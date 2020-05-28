from bluesky.traffic.asas import ConflictResolution
from bluesky.tools import geo
from bluesky.tools.aero import nm
import numpy as np
from bluesky.tools.aero import ft, nm

# Try to import pyclipper
try:
    import pyclipper
except ImportError:
    print("Could not import pyclipper, RESO SSD will not function")

# weight values for the cost function
WEIGHT_CHANGE_SPEED = 0.4
WEIGHT_CHANGE_PATH = 0.4
WEIGHT_DISTANCE_INTRUDERS = 0.2
PENALTY_LOS_MAX = 200

# discretize possible heading, speed changes
SPEED_STEPS = 5
HEADING_STEPS = 20


def init_plugin():
    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name': 'cost',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type': 'sim'
    }

    # init_plugin() should always return these two dicts.
    return config, {}


class cost(ConflictResolution):
    def setprio(self, flag=None, priocode=''):
        return super().setprio(flag, priocode)

    def detect_future_conflicts(self, rpz, hpz, dtlookahead, ntraf, lat, lon, gs, trk, alt, vs):
        ''' Conflict detection between ownship (traf) and intruder (traf/adsb).'''

        # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
        I = np.eye(ntraf)

        # Horizontal conflict ------------------------------------------------------

        # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(lat), np.asmatrix(lon),
                                           np.asmatrix(lat), np.asmatrix(lon))

        # Convert back to array to allow element-wise array multiplications later on
        # Convert to meters and add large value to own/own pairs
        qdr = np.asarray(qdr)
        dist = np.asarray(dist) * nm + 1e9 * I

        # Calculate horizontal closest point of approach (CPA)
        qdrrad = np.radians(qdr)
        dx = dist * np.sin(qdrrad)  # is pos j rel to i
        dy = dist * np.cos(qdrrad)  # is pos j rel to i

        # Ownship track angle and speed
        owntrkrad = np.radians(trk)
        ownu = gs * np.sin(owntrkrad).reshape((1, ntraf))  # m/s
        ownv = gs * np.cos(owntrkrad).reshape((1, ntraf))  # m/s

        # Intruder track angle and speed
        inttrkrad = np.radians(trk)
        intu = gs * np.sin(inttrkrad).reshape((1, ntraf))  # m/s
        intv = gs * np.cos(inttrkrad).reshape((1, ntraf))  # m/s

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
        dalt = alt.reshape((1, ntraf)) - \
               alt.reshape((1, ntraf)).T + 1e9 * I

        dvs = vs.reshape(1, ntraf) - \
              vs.reshape(1, ntraf).T
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

        return np.sqrt(dcpa2), swconfl

    def detect(self, asas, traf):
        """ Detect all current conflicts """

        # Check if ASAS is ON first!
        if not asas.swasas:
            return

    def resolve(self, conf, ownship, intruder):
        # it is assumed all aircraft are the same type
        self.heading_path = np.array([])
        self.speed_path = np.array([])

        # Now assign resolutions to variables in the ASAS class
        # Start with current states, need a copy, otherwise it changes traf!
        self.new_track = np.copy(ownship.hdg)
        self.new_gs = np.copy(ownship.gs)

        new_value = 0
        for i in range(HEADING_STEPS):
            self.heading_path = np.append(self.heading_path, new_value)
            new_value += 360 / HEADING_STEPS

        new_value = 0
        for i in range(SPEED_STEPS + 1):
            self.speed_path = np.append(self.speed_path, new_value)
            new_value += (ownship.perf.vmax[0] - 0) / SPEED_STEPS

        self.calculate_resolution(conf, ownship)

        # Not needed as it is a 2D-implementation...
        newvs = ownship.vs

        # Cap the velocity
        newgscapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, self.new_gs))

        alt = ownship.selalt

        return self.new_track, newgscapped, newvs, alt

    def get_route_heading_speed(self, aircraft_id, traf):
        routes = np.array(traf.ap.route)
        route = routes[aircraft_id]
        aircraft_iactwp = route.iactwp
        if len(route.wplat) > 0:
            [desiredHeading, dummy] = geo.qdrdist(route.wplat[aircraft_iactwp - 1],
                                                  route.wplon[aircraft_iactwp - 1],
                                                  route.wplat[aircraft_iactwp],
                                                  route.wplon[aircraft_iactwp])
        desiredSpeed = route.wpspd[aircraft_iactwp]

        return desiredHeading, desiredSpeed

    def calculate_resolution(self, conf, traf):
        """ Calculates closest conflict-free point according to ruleset """

        # order aircraft by time to los
        id_aircraft_conf = []
        order = conf.tLOS.argsort()
        for i in order:
            id1, id2 = conf.confpairs[i]
            id1 = traf.id.index(id1)
            id2 = traf.id.index(id2)
            if id1 not in id_aircraft_conf:
                id_aircraft_conf.append(int(id1))
            if id1 not in id_aircraft_conf:
                id_aircraft_conf.append(int(id2))

        aicraft_traj_set = 0

        for it in id_aircraft_conf:

            # look first at the desired path/speed
            desired_heading, desired_speed = self.get_route_heading_speed(it, traf)

            heading_paths = np.append(self.heading_path, desired_heading)
            speed_paths = np.append(self.speed_path, desired_speed)

            # intruders are the aircraft not in conflict and the aicraft in conflicts that we've already analyzed
            intruders = np.append(np.arange(traf.ntraf)[conf.inconf == False], id_aircraft_conf[:aicraft_traj_set])

            # prepare data for matrix calculation
            intruders = np.append(intruders, it)
            intruders = intruders.astype(int)
            ntraf = len(intruders)

            lat = traf.lat[intruders]
            lon = traf.lon[intruders]
            gs = traf.gs[intruders]
            alt = traf.alt[intruders]
            vs = traf.vs[intruders]
            trk = traf.trk[intruders]

            best_cost = np.inf
            best_heading = desired_heading
            best_speed = desired_speed
            previous_distance_intruders = np.average(conf.dcpa) / nm

            for heading in heading_paths:
                for speed in speed_paths:

                    gs[-1] = speed
                    trk[-1] = heading

                    # distance to cpa
                    dcpa, swconfl = self.detect_future_conflicts(conf.rpz, conf.hpz, conf.dtlookahead,
                                                                 ntraf, lat, lon, gs, trk, alt, vs)
                    # consider only the aircraft we are analyzing
                    dcpa = dcpa[-1]
                    swconfl = swconfl[-1]

                    path_cost = WEIGHT_CHANGE_PATH * abs(heading - desired_heading) / HEADING_STEPS
                    speed_cost = WEIGHT_CHANGE_SPEED * abs(speed - desired_speed) / SPEED_STEPS
                    distance_intruders_cost = WEIGHT_DISTANCE_INTRUDERS * (
                                previous_distance_intruders - np.average(dcpa) / nm) / (conf.rpz * 10)
                    # this is only bad when distance is increased
                    distance_intruders_cost = max(distance_intruders_cost, 0)
                    intrusions_cost = 0
                    if np.count_nonzero(swconfl) > 0:
                        intrusion_severity = conf.rpz - (np.mean(dcpa[swconfl]))
                        intrusions_cost = PENALTY_LOS_MAX * intrusion_severity / conf.rpz

                    cost = path_cost + speed_cost + distance_intruders_cost + intrusions_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_heading = heading
                        best_speed = speed

            traf.trk[it] = best_heading
            traf.gs[it] = best_speed

            # Stores resolution vector
            self.new_track[it] =  best_heading
            self.new_gs[it] = best_speed

            aicraft_traj_set += 1
