import matplotlib.pyplot as plt
import numpy as np
import torch


class FaceDrawer:
    def __init__(self):
        self.base_face = lambda: [
            (plt.gca().add_patch(plt.Circle((0.5, 0.5), 0.5, fc=(.9, .9, 0))),
             plt.gca().add_patch(plt.Circle((0.25, 0.6), 0.1, fc=(0, 0, 0))),
             plt.gca().add_patch(plt.Circle((0.75, 0.6), 0.1, fc=(0, 0, 0))))
        ]

        self.patches = {
            'smile': lambda: plt.gca().add_patch(
                plt.Polygon(np.stack([np.linspace(0.2, 0.8), 0.3 - np.sin(np.linspace(0, 3.14)) * 0.15]).T,
                            closed=False, fill=False, lw=3)),
            'frown': lambda: plt.gca().add_patch(
                plt.Polygon(np.stack([np.linspace(0.2, 0.8), 0.15 + np.sin(np.linspace(0, 3.14)) * 0.15]).T,
                            closed=False, fill=False, lw=3)),
            'left_eb_down': lambda: plt.gca().add_line(
                plt.Line2D([0.15, 0.35], [0.75, 0.7], color=(0, 0, 0))),
            'right_eb_down': lambda: plt.gca().add_line(
                plt.Line2D([0.65, 0.85], [0.7, 0.75], color=(0, 0, 0))),
            'left_eb_up': lambda: plt.gca().add_line(
                plt.Line2D([0.15, 0.35], [0.7, 0.75], color=(0, 0, 0))),
            'right_eb_up': lambda: plt.gca().add_line(
                plt.Line2D([0.65, 0.85], [0.75, 0.7], color=(0, 0, 0))),
        }

        self.sorted_keys = sorted(self.patches.keys())

    def draw_face(self, face):
        self.base_face()
        for i in face:
            self.patches[i]()
        plt.axis('scaled')
        plt.axis('off')

    def draw_sample_faces(self):
        faces_to_draw = [['smile', 'left_eb_down', 'right_eb_down'],
                         ['frown', 'left_eb_up', 'right_eb_up']]
        f, ax = plt.subplots(1, 2)
        plt.sca(ax[0])
        self.draw_face(faces_to_draw[0])
        plt.sca(ax[1])
        self.draw_face(faces_to_draw[1])
        plt.show()


class Tools(FaceDrawer):
    # super().__init__()

    def face_to_tensor(self, face):
        return torch.tensor([i in face for i in self.sorted_keys]).float()

    def face_parents(self, state):
        parent_states = []  # states that are parents of state
        parent_actions = []  # actions that lead from those parents to state
        for face_part in state:
            # For each face part, there is a parent without that part
            parent_states.append([i for i in state if i != face_part])
            # The action to get there is the corresponding index of that face part
            parent_actions.append(self.sorted_keys.index(face_part))
        return parent_states, parent_actions

    @staticmethod
    def plot_figure(loss_arr):
        plt.figure(figsize=(10, 3))
        plt.plot(loss_arr)
        plt.yscale('log')
        plt.show()


class RewardCalc:
    eyebrows = ('left_eb_down', 'left_eb_up', 'right_eb_down', 'right_eb_up')
    @staticmethod
    def has_overlap(face):
        # Can't have two overlapping eyebrows!
        if 'left_eb_down' in face and 'left_eb_up' in face:
            return True
        if 'right_eb_down' in face and 'right_eb_up' in face:
            return True
        # Can't have two overlapping mouths!
        if 'smile' in face and 'frown' in face:
            return True
        return False

    def face_reward(self, face):
        if self.has_overlap(face):
            return 0
        # Must have exactly two eyebrows
        if sum([i in face for i in self.eyebrows]) != 2:
            return 0
        # We want twice as many happy faces as sad faces so here we give a reward of 2 for smiles
        if 'smile' in face:
            return 2
        if 'frown' in face:
            return 1  # and a reward of 1 for frowns
        # If we reach this point, there's no mouth
        return 0

    def is_valid_face(self, face):
        # A valid face has exactly two eyebrows and no overlapping features
        return sum([i in face for i in self.eyebrows]) == 2 and not self.has_overlap(face)

    def is_smily_face(self, face):
        # A smiley face has a smile and is a valid face
        return 'smile' in face and self.is_valid_face(face)


def main():
    face_drawer = FaceDrawer()
    face_drawer.draw_sample_faces()


if __name__ == "__main__":
    main()
