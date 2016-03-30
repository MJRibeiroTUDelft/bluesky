try:
    # Try Qt5 first
    from PyQt5.QtCore import QThread, QObject
except ImportError:
    # Else PyQt4 imports
    from PyQt4.QtCore import QThread, QObject
import time

# Local imports
from screenio import ScreenIO
from simevents import StackTextEventType, BatchEventType, BatchEvent, SimStateEvent, SimQuitEventType
from ...traf import Traffic
from ...stack import Commandstack
# from ...traf import Metric
from ... import settings
from ...tools.datafeed import Modesbeast


class Simulation(QObject):
    # =========================================================================
    # Settings
    # =========================================================================
    # Simulation timestep [seconds]
    simdt = settings.simdt

    # Simulation loop update rate [Hz]
    sys_rate = settings.sim_update_rate

    # simulation modes
    init, op, hold, end = range(4)

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self, manager, navdb):
        super(Simulation, self).__init__()
        self.manager     = manager
        self.running     = True
        self.mode        = Simulation.init
        self.samplecount = 0
        self.sysdt       = 1000 / self.sys_rate

        # Set starting system time [milliseconds]
        self.syst        = 0.0

        # Starting simulation time [seconds]
        self.simt        = 0.0

        # Flag indicating running at fixed rate or fast time
        self.ffmode      = False
        self.ffstop      = None

        # Simulation objects
        self.screenio    = ScreenIO(self, manager)
        self.traf        = Traffic(navdb)
        self.stack       = Commandstack(self, self.traf, self.screenio)
        self.navdb       = navdb
        # Metrics
        self.metric      = None
        # self.metric      = Metric()
        self.beastfeed     = Modesbeast(self.stack, self.traf)

    def doWork(self):
        self.syst = int(time.time() * 1000.0)
        self.fixdt = self.simdt
        self.sendState()

        while self.running:
            # Timing bookkeeping
            self.samplecount += 1

            # Update the Mode-S beast parsing
            self.beastfeed.update()

            # TODO: what to do with init
            if self.mode == Simulation.init:
                self.mode = Simulation.op

            if self.mode == Simulation.op:
                self.stack.checkfile(self.simt)

            # Always update stack
            self.stack.process(self, self.traf, self.screenio)

            if self.mode == Simulation.op:
                self.traf.update(self.simt, self.simdt)

                # Update metrics
                if self.metric is not None:
                    self.metric.update(self, self.traf)

                # Update time for the next timestep
                self.simt += self.simdt

            # Process Qt events
            self.manager.processEvents()

            # When running at a fixed rate, increment system time with sysdt and calculate remainder to sleep
            if not self.ffmode:
                self.syst += self.sysdt
                remainder = self.syst - int(1000.0 * time.time())

                if remainder > 0:
                    QThread.msleep(remainder)
            elif self.ffstop is not None and self.simt >= self.ffstop:
                self.start()

    def stop(self):
        self.mode = Simulation.end
        self.sendState()

    def start(self):
        if self.ffmode:
            self.syst = int(time.time() * 1000.0)
        self.ffmode = False
        self.mode   = self.op

    def pause(self):
        self.mode   = self.hold

    def reset(self):
        self.simt   = 0.0
        self.mode   = self.init
        self.traf.reset(self.navdb)

    def fastforward(self, nsec=None):
        self.ffmode = True
        if nsec is not None:
            self.ffstop = self.simt + nsec
        else:
            self.ffstop = None

    def datafeed(self, flag):
        if flag == "ON":
            self.beastfeed.connectToHost(settings.modeS_host,
                                         settings.modeS_port)
        if flag == "OFF":
            self.beastfeed.disconnectFromHost()

    def setScenName(self, name):
        self.screenio.echo('Starting scenario' + name)

    def sendState(self):
        self.manager.sendEvent(SimStateEvent(self.mode))

    def batch(self, filename):
        # The contents of the scenario file are meant as a batch list: send to manager and clear stack
        self.stack.openfile(filename)
        self.manager.sendEvent(BatchEvent(self.stack.scentime, self.stack.scencmd))
        self.stack.scentime = []
        self.stack.scencmd  = []

    def event(self, event):
        # Keep track of event processing
        event_processed = False

        if event.type() == StackTextEventType:
            # We received a single stack command. Add it to the existing stack
            self.stack.stack(event.cmdtext)
            event_processed = True

        elif event.type() == BatchEventType:
            # We are in a batch simulation, and received an entire scenario. Assign it to the stack.
            self.stack.scentime = event.scentime
            self.stack.scencmd  = event.scencmd
            event_processed     = True
        elif event.type() == SimQuitEventType:
            # BlueSky is quitting
            self.running = False
        else:
            # This is either an unknown event or a gui event.
            event_processed = self.screenio.event(event)

        return event_processed
