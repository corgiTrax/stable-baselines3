import random
import sys

class FeedbackInterface():
    def __init__(self, parent=None):
        pass

    def initUI(self):
        pass

    def createGridLayout(self):
        pass

    def btnstate(self, btn):
        pass

    def updateLoss(self, loss_val):
        pass

    def updateMouseX(self, mouseX):
        pass

    def updateMouseY(self, mouseY):
        pass

    def updateSelectedObject(self, selected_object):
        pass

    def updateReward(self, reward_val):
        pass

    def updateHumanReward(self, reward_val):
        pass

    def updateStateEstimation(self, state):
        pass

    def updateHumanSignal(self, signal):
        pass

    def updateBDDL(self, bddl_val):
        pass

    def textchangedX(self, text):
        pass

    def textchangedY(self, text):
        pass

    def textchangedZ(self, text):
        pass

    def textChangedLoadModel(self, text):
        pass
    
    def update_font(self, key):
        pass


def main():
    ex = FeedbackInterface()


if __name__ == "__main__":
    main()
