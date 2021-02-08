# further developed by Jannik Drögemüller, Mats Haring, Franziska Schmidt and Simon Koschel

import random



class Task:

    def __init__(self, taskid, period, deadline, execution, prob = 0):
        self.id = taskid
        self.period = period
        self.deadline = deadline
        self.execution = execution
        self.abnormal_exe = execution
        self.activeJobs = []
        # 0 is the highest priority
        self.priority = -1
        self.processor = -1
        self.prob = prob


    def __getitem__(self,key):
        return getattr(self,key)

    class Job(object):
        def __init__(self, deadline, workload):
            self.deadline = deadline
            self.workload = workload

        def __str__(self):
            return "deadline: " + str(self.deadline) + ", workload: " + str(self.workload)
        
        def __repr__(self):
            return "deadline: " + str(self.deadline) + ", workload: " + str(self.workload)

    def earliestDeadline(self):
        if len(self.activeJobs) > 0:
            return self.activeJobs[0].deadline
        else:
            return None

    def lastDeadline(self):
        if len(self.activeJobs) > 0:
            return self.activeJobs[-1].deadline
        else:
            return None

    def workload(self):
        wl = 0
        for job in self.activeJobs:
            wl += job.workload
        return wl

    def updateWorkload(self, delta):
        if self.workload() > delta:
            print("BUG: Actual workload of Task " + str(self.id) + " is less than " + str(delta) + ".")
        else:
            while delta > 0 and len(self.activeJobs) > 0:
                job = self.activeJobs[0]
                if delta >= job.workload:
                    delta -= job.workload
                    self.activeJobs.pop(0)
                else:
                    job.workload -= delta
                    delta = 0

    def addJob(self, faultRate, currentTime):
        if faultRate == 0:
            self.activeJobs.append(self.Job(self.deadline + currentTime, self.execution))
        elif faultRate == 1:
            self.activeJobs.append(self.Job(self.deadline + currentTime, self.abnormal_exe))
        else:
            if random.randint(0,int(1/faultRate)-1) > int(1/faultRate)-2:
                self.activeJobs.append(self.Job(self.deadline + currentTime, self.abnormal_exe))
            else:
                self.activeJobs.append(self.Job(self.deadline + currentTime, self.execution))

    def updateDeadline(self, delta):
        for job in self.activeJobs:
            job.deadline -= delta

    def setAbnormalExe(self, abnormal_exe):
        self.abnormal_exe = abnormal_exe

    def setPriority(self, prio):
        self.priority = prio

    def setProcessor(self, processor):
        self.processor = processor

    def convertToArr(self):
        return [self.id, self.period, self.deadline, self.execution, self.abnormal_exe, self.priority, self.processor, self.prob]

    def __str__(self):
        return "id: " + str(self.id) + ", period: " + str(self.period) + ", deadline: " + str(self.deadline) + ", execution: " + str(self.execution) + ", priority: " + str(self.priority) + ", processor: " + str(self.processor) + ", jobs: " + str(self.activeJobs)
        
    def __repr__(self):
        return "id: " + str(self.id) + ", period: " + str(self.period) + ", deadline: " + str(self.deadline) + ", execution: " + str(self.execution) + ", priority: " + str(self.priority) + ", processor: " + str(self.processor) + ", jobs: " + str(self.activeJobs)