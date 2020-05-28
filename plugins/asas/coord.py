from bluesky.traffic.asas import ConflictResolution
from bluesky.tools import geo
from bluesky.tools.aero import nm
import numpy as np
import random

# Try to import pyclipper
try:
    import pyclipper
except ImportError:
    print("Could not import pyclipper, RESO SSD will not function")

MAX_IT = 5
PREFERENCES = []


def init_plugin():
    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'coord',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim'
    }

    # init_plugin() should always return these two dicts.
    return config, {}


class coord(ConflictResolution):
    def setprio(self, flag=None, priocode=''):
        return super().setprio(flag, priocode)

    def qdrdist_matrix_indices(self, ntraf):
        """ This function gives the indices that can be used in the lon/lat-vectors """
        # The indices will be n*(n-1)/2 long
        # Only works for n >= 2, which is logical...
        # This is faster than np.triu_indices :)
        tmp_range = np.arange(ntraf - 1, dtype=np.int32)
        ind1 = np.repeat(tmp_range, (tmp_range + 1)[::-1])
        ind2 = np.ones(ind1.shape[0], dtype=np.int32)
        inds = np.cumsum(tmp_range[1:][::-1] + 1)
        np.put(ind2, inds, np.arange(ntraf * -1 + 3, 1))
        ind2 = np.cumsum(ind2, out=ind2)
        return ind1, ind2

    def detect_conflicts(self, ownship, intruder, rpz, hpz, dtlookahead):
        ''' Conflict detection between ownship (traf) and intruder (traf/adsb).'''
        # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
        I = np.eye(ownship.ntraf)

        # Horizontal conflict ------------------------------------------------------

        # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(ownship.lat), np.asmatrix(ownship.lon),
                                           np.asmatrix(intruder.lat), np.asmatrix(intruder.lon))

        # Convert back to array to allow element-wise array multiplications later on
        # Convert to meters and add large value to own/own pairs
        qdr = np.asarray(qdr)
        dist = np.asarray(dist) * nm + 1e9 * I

        # Calculate horizontal closest point of approach (CPA)
        qdrrad = np.radians(qdr)
        dx = dist * np.sin(qdrrad)  # is pos j rel to i
        dy = dist * np.cos(qdrrad)  # is pos j rel to i

        # Ownship track angle and speed
        owntrkrad = np.radians(ownship.trk)
        ownu = ownship.gs * np.sin(owntrkrad).reshape((1, ownship.ntraf))  # m/s
        ownv = ownship.gs * np.cos(owntrkrad).reshape((1, ownship.ntraf))  # m/s

        # Intruder track angle and speed
        inttrkrad = np.radians(intruder.trk)
        intu = intruder.gs * np.sin(inttrkrad).reshape((1, ownship.ntraf))  # m/s
        intv = intruder.gs * np.cos(inttrkrad).reshape((1, ownship.ntraf))  # m/s

        du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
        dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

        dv2 = du * du + dv * dv
        dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
        vrel = np.sqrt(dv2)

        tcpa = -(du * dx + dv * dy) / dv2 + 1e9 * I

        # Calculate distance^2 at CPA (minimum distance^2)
        dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

        # Check for horizontal conflict
        R2 = rpz * rpz
        swhorconf = dcpa2 < R2  # conflict or not

        # Calculate times of entering and leaving horizontal conflict
        dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
        dtinhor = dxinhor / vrel

        tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
        touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf

        # Vertical conflict --------------------------------------------------------

        # Vertical crossing of disk (-dh,+dh)
        dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
               intruder.alt.reshape((1, ownship.ntraf)).T + 1e9 * I

        dvs = ownship.vs.reshape(1, ownship.ntraf) - \
              intruder.vs.reshape(1, ownship.ntraf).T
        dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero

        # Check for passing through each others zone
        tcrosshi = (dalt + hpz) / -dvs
        tcrosslo = (dalt - hpz) / -dvs
        tinver = np.minimum(tcrosshi, tcrosslo)
        toutver = np.maximum(tcrosshi, tcrosslo)

        # Combine vertical and horizontal conflict----------------------------------
        tinconf = np.maximum(tinver, tinhor)
        toutconf = np.minimum(toutver, touthor)

        swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) * \
                           (tinconf < dtlookahead) * (1.0 - I), dtype=np.bool)

        # --------------------------------------------------------------------------
        # Update conflict lists
        # --------------------------------------------------------------------------
        # Ownship conflict flag and max tCPA
        inconf = np.any(swconfl, 1)

        return inconf, np.sqrt(dcpa2[swconfl])

    def detect(self, asas, traf):
        """ Detect all current conflicts """

        # Check if ASAS is ON first!
        if not asas.swasas:
            return

    def build_VOs(self, vmax, vmin, i_other, i, x1, x2, y1, y2, gseast, gsnorth, hdg, ind, preference, gs):
        """ Calculates the FRV and ARV of the SSD """

        # lets get several points, so we can get solutions as close as possible from the
        self.ARV = None

        # Discretize the circles using points on circle
        N_angle = 180
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
        # Put points of unit-circle in a (180x2)-array (CW)
        xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))

        # Map them into the format pyclipper wants. Outercircle CCW, innercircle CW
        circle_tup = (tuple(map(tuple, np.flipud(xyc * vmax))), tuple(map(tuple, xyc * vmin)))

        # VO from 2 to 1 is mirror of 1 to 2. Only 1 to 2 can be constructed in
        # this manner, so need a correction vector that will mirror the VO
        fix = np.ones(np.shape(i_other))
        fix[i_other < i] = -1
        # Relative bearing [deg] from [-180,180]
        # (less required conversions than rad in RotA)
        fix_ang = np.zeros(np.shape(i_other))
        fix_ang[i_other < i] = 180.

        # Get vertices in an x- and y-array of size (ntraf-1)*3x1
        x = np.concatenate((gseast[i_other],
                            x1[ind] * fix + gseast[i_other],
                            x2[ind] * fix + gseast[i_other]))
        y = np.concatenate((gsnorth[i_other],
                            y1[ind] * fix + gsnorth[i_other],
                            y2[ind] * fix + gsnorth[i_other]))
        # Reshape [(ntraf-1)x3] and put arrays in one array [(ntraf-1)x3x2]
        x = np.transpose(x.reshape(3, np.shape(i_other)[0]))
        y = np.transpose(y.reshape(3, np.shape(i_other)[0]))
        xy = np.dstack((x, y))

        # Make a clipper object
        pc = pyclipper.Pyclipper()
        # Add circles (ring-shape) to clipper as subject
        pc.AddPaths(pyclipper.scale_to_clipper(circle_tup), pyclipper.PT_SUBJECT, True)

        # Add each other other aircraft to clipper as clip
        for j in range(np.shape(i_other)[0]):
            VO = pyclipper.scale_to_clipper(tuple(map(tuple, xy[j, :, :])))
            pc.AddPath(VO, pyclipper.PT_CLIP, True)

        ARV = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Make another clipper object for extra intersections
        pc2 = pyclipper.Pyclipper()
        # Put the ARV_aux in there
        pc2.AddPaths(ARV, pyclipper.PT_CLIP, True)

        # For resolution purposes sometimes extra intersections are wanted
        if preference == 0:
            # preference for heading change first
            xyp = (tuple(map(tuple, np.flipud(xyc * min(vmax, gs[i] + 0.1)))),
                   tuple(map(tuple, xyc * max(vmin, gs[i] - 0.1))))
            part = pyclipper.scale_to_clipper(xyp)
            pc.AddPaths(part, pyclipper.PT_SUBJECT, True)
        else:
            # preference for speed change first
            hdg_sel = hdg[i] * np.pi / 180
            xyp = np.array([[np.sin(hdg_sel - 0.0087), np.cos(hdg_sel - 0.0087)],
                            [0, 0],
                            [np.sin(hdg_sel + 0.0087), np.cos(hdg_sel + 0.0087)]],
                           dtype=np.float64)
            part = pyclipper.scale_to_clipper(tuple(map(tuple, 1.1 * vmax * xyp)))
            pc.AddPath(part, pyclipper.PT_SUBJECT, True)
        # Execute clipper command
        self.ARV = pyclipper.scale_from_clipper(
            pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))

    def calculate_avoidance_maneuver(self, gseast, gsnorth, it, previous_solutions):

        if len(self.ARV) > 0:
            # Loop through all exteriors and append. Afterwards concatenate
            p = []
            q = []
            for j in range(len(self.ARV)):
                p.append(np.array(self.ARV[j]))
                q.append(np.diff(np.row_stack((p[j], p[j][0])), axis=0))
            p = np.concatenate(p)
            q = np.concatenate(q)
            # Calculate squared distance between edges
            l2 = np.sum(q ** 2, axis=1)
            # Catch l2 == 0 (exception)
            same = l2 < 1e-8
            l2[same] = 1.
            # Calc t
            t = np.sum((np.array([gseast[it], gsnorth[it]]) - p) * q, axis=1) / l2
            # Speed of boolean indices only slightly faster (negligible)
            # t must be limited between 0 and 1
            t = np.clip(t, 0., 1.)
            t[same] = 0.
            # Calculate closest point to each edge
            x1 = p[:, 0] + t * q[:, 0]
            y1 = p[:, 1] + t * q[:, 1]

            # Get distance squared
            # gseast and gsnort are updated after the first cycle
            # so in the following cycles we will be looking for solutions
            # close to the previous solution
            d2 = (x1 - gseast[it]) ** 2 + (y1 - gsnorth[it]) ** 2

            # Sort distance
            ind = np.argsort(d2)
            x1 = x1[ind]
            y1 = y1[ind]

            it = 0
            # do not the repeat previous solutions, there is no improvement there
            break_cycle = False
            while not break_cycle:
                break_cycle = True
                for previous_solution in previous_solutions:
                    if abs(x1[it] - previous_solution[0]) < 1e-10 and abs(y1[it] - previous_solution[1]) < 1e-10:
                        it += 1
                        break_cycle = False
                        continue

            return x1[min(it, len(x1 - 1))], y1[min(it, len(y1 - 1))]
        else:
            return 0, 0

    def calculate_best_maneuver(self, vmax, vmin, i_other, i, x1, x2, y1, y2, gseast, gsnorth, gs, hdg, ind,
                                preference, previous_solutions):

        self.build_VOs(vmax, vmin, i_other, i, x1, x2, y1, y2, gseast, gsnorth, hdg, ind, preference, gs)
        return self.calculate_avoidance_maneuver(gseast, gsnorth, i, previous_solutions)

    def resolve(self, conf, ownship, intruder):
        # it is assumed all aircraft are the same type
        self.heading_path = np.array([])
        self.speed_path = np.array([])

        # assume all aircraft are the same type
        self.vmin = max(ownship.perf.vmin[0], 0)
        self.vmax = ownship.perf.vmax[0]

        while len(PREFERENCES) < ownship.ntraf:
            PREFERENCES.append(random.randint(0, 1))

        # Now assign resolutions to variables in the ASAS class
        # Start with current states, need a copy, otherwise it changes traf!
        self.new_track = np.copy(ownship.hdg)
        self.new_gs = np.copy(ownship.gs)

        # in the first time step, ASAS runs before perf, which means that his value will be zero
        if self.vmin != self.vmax:
            self.calculate_resolution(conf, ownship)

        # Not needed as it is a 2D-implementation...
        newvs = ownship.vs

        # Cap the velocity
        newgscapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, self.new_gs))

        alt = ownship.selalt

        return self.new_track, newgscapped, newvs, alt

    def calculate_resolution(self, conf, traf):
        """ Calculates closest conflict-free point according to ruleset """

        ind1, ind2 = self.qdrdist_matrix_indices(traf.ntraf)
        [qdr, dist] = geo.qdrdist_matrix(traf.lat[ind1], traf.lon[ind1], traf.lat[ind2], traf.lon[ind2])
        qdr = np.reshape(np.array(qdr), np.shape(ind1))
        dist = np.reshape(np.array(dist), np.shape(ind1))
        # SI-units from [deg] to [rad]
        qdr = np.deg2rad(qdr)
        # Get distance from [nm] to [m]
        dist = dist * nm

        hsepm = conf.rpz * self.resofach  # [m] Horizontal separation with safety margin
        # In LoS the VO can't be defined, act as if dist is on edge
        dist[dist < hsepm] = hsepm

        self.inconf = np.copy(conf.inconf)

        # Calculate vertices of Velocity Obstacle (CCW)
        # These are still in relative velocity space, see derivation in appendix
        # Half-angle of the Velocity obstacle [rad]
        # Include safety margin
        alpha = np.arcsin(hsepm / dist)
        # Limit half-angle alpha to 89.982 deg. Ensures that VO can be constructed
        alpham = 0.4999 * np.pi
        alpha[alpha > alpham] = alpham
        # Relevant sin/cos/tan
        sinqdr = np.sin(qdr)
        cosqdr = np.cos(qdr)
        tanalpha = np.tan(alpha)
        cosqdrtanalpha = cosqdr * tanalpha
        sinqdrtanalpha = sinqdr * tanalpha

        # Relevant x1,y1,x2,y2 (x0 and y0 are zero in relative velocity space)
        x1 = (sinqdr + cosqdrtanalpha) * 2 * self.vmax
        x2 = (sinqdr - cosqdrtanalpha) * 2 * self.vmax
        y1 = (cosqdr - sinqdrtanalpha) * 2 * self.vmax
        y2 = (cosqdr + sinqdrtanalpha) * 2 * self.vmin

        previous_solutions = dict()

        for cycle in range(MAX_IT):

            # Consider every aircraft
            for it in range(traf.ntraf):

                if previous_solutions.get(it) is None:
                    previous_solutions[it] = []

                # Calculate solution for aircraft in conflict
                if self.inconf[it]:
                    ind1, ind2 = self.qdrdist_matrix_indices(traf.ntraf)
                    # Get indices that belong to aircraft i
                    ind = np.where(np.logical_or(ind1 == it, ind2 == it))[0]

                    # other traffic
                    i_other = np.delete(np.arange(0, traf.ntraf), it)

                    # find new solution
                    asase, asasn = self.calculate_best_maneuver(self.vmax, self.vmin, i_other,
                                                                it, x1, x2, y1, y2, traf.gseast, traf.gsnorth, traf.gs,
                                                                traf.trk, ind, PREFERENCES[it], previous_solutions[it])

                    previous_solutions[it].append([asase, asasn])

                    self.new_track[it] = np.arctan2(asase, asasn) * 180 / np.pi
                    self.new_gs[it] = np.sqrt(asase ** 2 + asasn ** 2)

            traf.trk = np.copy(self.new_track)
            traf.gs = np.copy(self.new_gs)

            traf.gsnorth = np.cos(traf.trk / 180 * np.pi) * traf.gs
            traf.gseast = np.sin(traf.trk / 180 * np.pi) * traf.gs

            # the negotiations is over for aircraft that are no longer in conflict
            self.inconf, dist = self.detect_conflicts(traf, traf, conf.rpz, conf.hpz, conf.dtlookahead)
