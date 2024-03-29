# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels

from environments.wrappers import (
    ActionDTypeWrapper, ActionRepeatWrapper, ExtendedTimeStepWrapper,
    FrameStackWrapper, NoisyMaskWrapper, SegmentationToRobotMaskWrapper,
    SlimMaskWrapper, StackRGBAndMaskWrapper)

name2robot_seg_ids = {
    "cup_catch": list(range(1, 7)),
    "cartpole_swingup": list(range(3, 5)),
    "cheetah_run": list(range(1, 9)),
    "finger_spin": list(range(1, 5)),
    "reacher_easy": [5, 7, 8, 9],
    "walker_walk": list(range(1, 8))
}

# from sear.environments.wrappers import (ActionDTypeWrapper,
#                                              ActionRepeatWrapper,
#                                              ExtendedTimeStepWrapper,
#                                              FrameStackWrapper)


def make(name, frame_stack, action_repeat, seed, add_segmentation_to_obs):
    domain, task = name.split("_", 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup="ball_in_cup").get(domain, domain)
    pixels_keys = []
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={"random": seed},
                         visualize_reward=False)
        pixels_key = "pixels"
    else:
        name = f"{domain}_{task}_vision"
        env = manipulation.load(name, seed=seed)
        pixels_key = "front_close"
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)

    pixels_keys.append(pixels_key)
    if add_segmentation_to_obs:
        segmentation_key = "segmentation"
        pixels_keys.append(segmentation_key)
        segmentation_kwargs = dict(height=84,
                                   width=84,
                                   camera_id=camera_id,
                                   segmentation=True)
        env = pixels.Wrapper(env,
                             pixels_only=False,
                             render_kwargs=segmentation_kwargs,
                             observation_key=segmentation_key)
        env.robot_segmentation_ids = name2robot_seg_ids[name]
        env = SegmentationToRobotMaskWrapper(
            env,
            segmentation_key,
            types_channel=1,
        )

    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_keys)
    env = ExtendedTimeStepWrapper(env)
    return env
