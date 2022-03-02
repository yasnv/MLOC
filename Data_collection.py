import gym
import pygame
import matplotlib
import argparse
from gym import logger
import numpy as np

from collections import deque
from pygame.locals import VIDEORESIZE


def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(
        arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
    I = I[103:
          175]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2, ::2, 0]  # downsample by factor of 2.
    I[I == 180] = 0  # erase background (background type 1)
    # I[I == 109] = 0  # erase background (background type 2)
    # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    I[I != 0] = 1
    # ravel flattens an array and collapses it into a column vector
    return I.astype(np.float).ravel()


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    env.reset()
    rendered = env.render(mode="rgb_array")

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, (
                env.spec.id
                + " does not have explicit key to action mapping, "
                + "please specify one manually"
            )
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()
    fileObs = open('Obs2.csv', 'w')  # write to file
    filerewact = open('revact2.csv', 'w')  # write to file
    filelabel = open('label2.csv', 'w')  # write to file
    episodes = 1
    while running:
        if env_done:
            env_done = False
            obs = env.reset()
            prep1 = prepro(obs)
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)
            prep = prepro(obs)
            prep2 = prep-prep1 if prep1 is not None else np.zeros(prep2)
            prep1 = prep
            for i in range(2880):
                fileObs.write(str(prep2[i]))
                fileObs.write(',')
            fileObs.write('\n')
            filelabel.write(str(episodes))
            filelabel.write('\n')
            filerewact.write(str(action))
            filerewact.write(',')
            filerewact.write(str(rew))
            filerewact.write('\n')
            if env_done:
                episodes = episodes+1
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
        if obs is not None:
            rendered = env.render(mode="rgb_array")
            display_arr(screen, rendered, transpose=transpose,
                        video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="MontezumaRevengeNoFrameskip-v4",
        help="Define Environment",
    )
    args = parser.parse_args()
    env = gym.make('ALE/Bowling-v5')
    play(env, zoom=4, fps=60)


if __name__ == "__main__":
    main()
