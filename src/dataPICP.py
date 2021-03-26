

class dataPICP:
    def __init__(self, q_opt=None, path=None, distancesSE3=None):
        self.path=path
        self.distancesSE3=distancesSE3
        self.q_opt=q_opt

    def append(self, otherData):
        lastTime = self.path["time"][-1]
        self.path["time"].extend([time + lastTime for time in otherData.path['time'][1:]])
        self.path["cost"].extend(otherData.path["cost"])
        self.path["pose"].extend(otherData.path["pose"])
        self.distancesSE3.extend(otherData.distancesSE3)
        self.q_opt = self.path['pose'][-1]

    def loadFromFile(self, name):
        self.path = {"time": [], "pose": [], "cost": []}
        self.distancesSE3 = []
        with open(name, "r") as file:
            file.readline()  # skip the headers
            for line in file:
                line_parsed = line.split(',')
                self.path["time"].append(float(line_parsed[0]))
                pose = {"pose_mean": []}
                for i in range(1, 7):
                    pose["pose_mean"].append(float(line_parsed[i]))
                self.path["pose"].append(pose)
                self.path["cost"].append(float(line_parsed[7]))
                self.distancesSE3.append(float(line_parsed[8]))
        self.q_opt = self.path['pose'][-1]

    @staticmethod
    def createFromFile(name):
        d = dataPICP()
        d.loadFromFile(name)
        return d

    def saveToFile(self, name):
        with open(name + ".txt", "w") as file:
            file.write("# time, pose, cost, distance \n")
            for time, pose, cost, dist in zip(self.path['time'], self.path['pose'], self.path['cost'], self.distancesSE3):
                stringToWrite = str(time) + ","
                for p in pose['pose_mean']:
                    stringToWrite += str(p) + ","
                stringToWrite += str(cost) + ","
                stringToWrite += str(dist)
                file.write(stringToWrite + "\n")
