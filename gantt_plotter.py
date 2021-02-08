# further developed by Jannik Drögemüller, Mats Haring, Franziska Schmidt and Simon Koschel

import plotly.figure_factory as ff

#df = [dict(Task="Job-1", Start='2017-01-01', Finish='2017-02-02', Resource='Complete'),
#      dict(Task="Job-1", Start='2017-02-15', Finish='2017-03-15', Resource='Incomplete'),
#      dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Not Started'),
#      dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Complete'),
#      dict(Task="Job-3", Start='2017-03-10', Finish='2017-03-20', Resource='Not Started'),
#      dict(Task="Job-3", Start='2017-04-01', Finish='2017-04-20', Resource='Not Started'),
#      dict(Task="Job-3", Start='2017-05-18', Finish='2017-06-18', Resource='Not Started'),
#      dict(Task="Job-4", Start='2017-01-14', Finish='2017-03-14', Resource='Complete')]
#
#colors = {'Not Started': 'rgb(220, 0, 0)',
#          'Incomplete': (1, 0.9, 0.16),
#          'Complete': 'rgb(0, 255, 100)'}
#fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
#                      group_tasks=True)
#fig.show()



class GanttPlotter:

    def __init__(self):
        self.data = []
        self.colors = dict(Empty ='#f58231',
                Task_0 = '#000000',
                Task_1 = '#ffe119',
                Task_2 = '#3cb44b',
                Task_3 = '#4363d8',
                Task_4 = '#e6194b',
                Task_5 = '#911eb4',
                Task_6 = '#46f0f0',
                Task_7 = '#f032e6',
                Task_8 = '#bcf60c',
                Task_9 = '#fabebe',
                Task_10 = '#008080',
                Task_11 = '#e6beff',
                Task_12 = '#9a6324',
                Task_13 = '#fffac8',
                Task_14 = '#800000',
                Task_15 = '#aaffc3',
                Task_16 = '#808000',
                Task_17 = '#ffd8b1',
                Task_18 = '#000075',
                Task_19 = '#808080',
                Task_20 = '#ffffff')
        self.highestTaskID = -1
        self.time = 0

    def plot(self):
        #itemlist = list(self.colors.items())
        #print(itemlist)
        #c = dict(itemlist[:self.highestTaskID+2])
        #print(c)
        fig = ff.create_gantt(self.data, colors=self.colors, index_col = 'Resource', show_colorbar = True, group_tasks = True)
        fig['layout']['xaxis'].update({'type': None})
        fig.show()
    
    def addDataLame(self, processor, start, finish, task):
        self.data.append(dict(Task="Processor " + str(processor), Start=str(start), Finish=str(finish), Resource='Task_' + str(task)))

    def addData(self, taskIDs, delta):
        for processor, taskID in enumerate(taskIDs):
            if taskID > self.highestTaskID:
                self.highestTaskID = taskID
            proc = 'Processor ' + str(processor)
            if taskID == -1:
                task = 'Empty'
            else:
                task = 'Task_' + str(taskID)
            self.data.append(dict(Task=proc, Start=self.time, Finish=self.time + delta, Resource=task))
        self.time += delta
        
