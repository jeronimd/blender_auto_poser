import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D

from poser.dataset import BaseDataset, BatchedDataset
from poser.forward import ForwardKinematicsLayer
from poser.geometry import (convert_matrix_to_quat, convert_ortho_to_quat,
                            convert_quat_to_matrix, convert_quat_to_ortho)
from utils.constants import Constants


class PoseVisualizer:
    def __init__(self):
        dataset = BaseDataset()
        self.dataset = BatchedDataset(dataset,
                                      batch_size=1,
                                      augment=True
                                      #   augment=False
                                      )
        self.offsets = torch.tensor([Constants.DEFAULT_BONE_OFFSETS_LOCAL_LOCATION[joint] for joint in Constants.BONE_IDX], dtype=torch.float32)

        self.current_idx = 0
        self.total_frames = len(dataset)
        self.bone_names = list(Constants.BONE_IDX.keys())

        # Initialize FK layer
        self.fk_layer = ForwardKinematicsLayer()

        # Setup figure and styling
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.set_facecolor('#f0f0f0')

        # Main 3D axis with styling
        self.ax = self.fig.add_axes([0.1, 0.25, 0.7, 0.7], projection='3d')
        self._style_axis(self.ax)

        # Control panel axis
        self.control_ax = self.fig.add_axes([0.82, 0.25, 0.16, 0.7])
        self.control_ax.set_facecolor('#ffffff')
        self._style_axis(self.control_ax, borders=True)

        # Navigation panel
        self.nav_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.15])
        self.nav_ax.set_facecolor('#ffffff')
        self._style_axis(self.nav_ax, borders=True)

        # Initialize visualization parameters
        self.elev = 15.0
        self.azim = 45.0
        self.zoom_level = 1.0
        self.anchor_point = np.array([0, 0, 1])  # Fixed anchor point

        # Create widgets
        self._create_controls()
        self.update_plot()

    def update_plot(self):
        self.ax.clear()

        # print("Frame:", self.current_idx)

        data = self.dataset[self.current_idx // self.dataset.batch_size]

        # Get ground truth data
        locations = data['joint_positions'][self.current_idx % self.dataset.batch_size].numpy()
        rotations = data['joint_rotations'][self.current_idx % self.dataset.batch_size].numpy()

        # locations_m = data['locations_m'].numpy()
        # rotations_m = data['rotations_m'].numpy()

        # Calculate FK positions
        with torch.no_grad():
            root_pos = torch.tensor(locations[0]).unsqueeze(0).float()

            rot_tensor = torch.tensor(rotations).unsqueeze(0).float()
            # rot_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, 65, 1)  # Should produce straight T-pose if offsets are correct

            fk_positions, fk_rotations = self.fk_layer(self.offsets, convert_quat_to_matrix(rot_tensor), root_pos)
            # fk_positions, fk_rotations = self.fk_layer(fk_rotations, root_pos)
            # fk_positions, _ = self.fk_layer(rot_tensor, root_pos)
            # fk_positions, _ = self.fk_layer(convert_quat_to_matrix(convert_ortho_to_quat(convert_quat_to_ortho(rot_tensor))), root_pos)  # Make sure the converts work
            fk_positions = fk_positions.squeeze(0).numpy()

            # print("Frame:", self.current_idx)
            # print("gt_rotations:", rot_tensor)
            # print("fk_rotations:", convert_matrix_to_quat(fk_rotations))
            # print("gt_locations:", locations)
            # print("fk_locations:", fk_positions)

            # # Calculate FK positions for mirrored data
            # root_pos_m = torch.tensor(locations_m[0]).unsqueeze(0).float()
            # rot_tensor_m = torch.tensor(rotations_m).unsqueeze(0).float()
            # fk_positions_m, _ = self.fk_layer(convert_quat_to_matrix(rot_tensor_m), root_pos_m)
            # fk_positions_m = fk_positions_m.squeeze(0).numpy()

        # # Print the GT and FK positions
        # for bone, idx in Constants.BONE_IDX.items():
        #     # Format each position vector with 5 decimal places
        #     gt_pos = [f"{x:.5f}" for x in gt_positions[idx]]
        #     fk_pos = [f"{x:.5f}" for x in fk_positions[idx]]
        #     print(f"{bone:<12}: LOC {gt_pos} ROT {fk_pos}")
        #     # print(f"{bone:<12}: GT {gt_pos} FK {fk_pos}")

        # Create comparison dictionary
        locations = {
            'gt_locations': (locations, '#3030ff'),
            'fk_locations': (fk_positions, '#ff3030'),
            # 'locations_m': (locations_m, '#78a5cf'),
            # 'fk_positions_m': (fk_positions_m, '#ffc430')
        }

        # Plot both versions
        for label, (pos, color) in locations.items():
            # Translate to anchor point
            pelvis_offset = pos[0] - self.anchor_point
            adjusted_pos = pos  # - pelvis_offset

            # Plot joints
            xs, ys, zs = adjusted_pos.T
            self.ax.scatter(xs, ys, zs, c=color, s=20, alpha=0.6, label=label)

            # Draw bone connections
            for bone, parent in Constants.BONE_CONNECTIONS.items():
                if parent is not None:
                    j_idx = Constants.BONE_IDX[bone]
                    p_idx = Constants.BONE_IDX[parent]
                    x = [adjusted_pos[p_idx][0], adjusted_pos[j_idx][0]]
                    y = [adjusted_pos[p_idx][1], adjusted_pos[j_idx][1]]
                    z = [adjusted_pos[p_idx][2], adjusted_pos[j_idx][2]]
                    self.ax.plot(x, y, z, color=color, linewidth=1.5, alpha=0.4)

        # Highlight input bones
        # input_indices = [Constants.BONE_IDX[name] for name in self.dataset.input_bones]
        # for idx in input_indices:
        #     x, y, z = adjusted_pos[idx]
        #     self.ax.text(x, y, z, self.bone_names[idx],
        #                  color='#c00000', fontsize=8, ha='center', va='bottom')

        # Set view parameters
        self._set_axis_limits()
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.ax.legend()
        self.ax.set_title(f'FK Validation - Frame {self.current_idx}', y=1.0)

        # Update status
        self.status_text.set_text(f"Frame: {self.current_idx + 1}/{self.total_frames}")
        self.fig.canvas.draw_idle()

    def _submit_frame(self, text):
        new_idx = int(text)
        if 0 <= new_idx < self.total_frames:
            self.current_idx = new_idx
            self.update_plot()

    def _prev_frame(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.text_box.set_val(str(self.current_idx))

    def _next_frame(self, event):
        if self.current_idx < self.total_frames - 1:
            self.current_idx += 1
            self.text_box.set_val(str(self.current_idx))

    def _reload_frame(self, event):
        self.update_plot()

    def _create_controls(self):
        # Navigation controls
        self.status_text = self.nav_ax.text(
            0.5, 0.7,
            f"Frame: {self.current_idx + 1}/{self.total_frames}",
            ha='center', va='center', fontsize=10)

        self.prev_btn = Button(
            plt.axes([0.3, 0.1, 0.1, 0.05]), 'Previous',
            color='#e0e0e0', hovercolor='#b0b0b0'
        )
        self.prev_btn.on_clicked(self._prev_frame)

        self.next_btn = Button(
            plt.axes([0.6, 0.1, 0.1, 0.05]), 'Next',
            color='#e0e0e0', hovercolor='#b0b0b0'
        )
        self.next_btn.on_clicked(self._next_frame)

        self.text_box = TextBox(
            plt.axes([0.45, 0.1, 0.1, 0.05]), 'Frame:',
            initial=str(self.current_idx),
            color='white', hovercolor='white'
        )
        self.text_box.on_submit(self._submit_frame)

        self.reload_btn = Button(
            plt.axes([0.75, 0.1, 0.1, 0.05]), 'Reload',
            color='#e0e0e0', hovercolor='#b0b0b0'
        )
        self.reload_btn.on_clicked(self._reload_frame)

        # View controls
        self.elev_slider = Slider(
            plt.axes([0.85, 0.7, 0.12, 0.04]),
            'Elevation', -90, 90, valinit=self.elev,
            color='#a0a0a0', track_color='#d0d0d0'
        )
        self.elev_slider.on_changed(self._update_view)

        self.azim_slider = Slider(
            plt.axes([0.85, 0.6, 0.12, 0.04]),
            'Azimuth', -180, 180, valinit=self.azim,
            color='#a0a0a0', track_color='#d0d0d0'
        )
        self.azim_slider.on_changed(self._update_view)

        self.zoom_slider = Slider(
            plt.axes([0.85, 0.5, 0.12, 0.04]),
            'Zoom', 0.1, 3.0, valinit=self.zoom_level,
            color='#a0a0a0', track_color='#d0d0d0'
        )
        self.zoom_slider.on_changed(self._update_zoom)

    def _update_view(self, val):
        self.elev = self.elev_slider.val
        self.azim = self.azim_slider.val
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.fig.canvas.draw_idle()

    def _update_zoom(self, val):
        self.zoom_level = self.zoom_slider.val
        self._set_axis_limits()
        self.fig.canvas.draw_idle()

    def _set_axis_limits(self):
        max_extent = 1.5 / self.zoom_level
        self.ax.set_xlim(-max_extent, max_extent)
        self.ax.set_ylim(-max_extent, max_extent)
        self.ax.set_zlim(1 - max_extent, 1 + max_extent)

    def _style_axis(self, ax, borders=False):
        if borders:
            for spine in ax.spines.values():
                spine.set_color('#808080')
                spine.set_linewidth(1)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_zticks([])


if __name__ == "__main__":
    visualizer = PoseVisualizer()
    plt.show()

    # python -m scripts.visualization
    # python -m scripts.visualization > training_output.log 2>&1
