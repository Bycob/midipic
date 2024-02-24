#!/usr/bin/python3

import sys
import os
import argparse
import logging

import mido
from mido import MidiFile, MidiTrack
import cv2 as cv

from create_dataset import img_to_midi


def main():
    parser = argparse.ArgumentParser(description="Listen to an image representing Midi")
    parser.add_argument("files", type=str, nargs='*', help="Input files")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--save", action="store_true", help="Save midi to a file instead of listening to it")
    parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")
    parser.add_argument(
        "--pixel_per_beat",
        type=int,
        default=4,
        help="Number of pixel (vertical) in a beat",
    )
    parser.add_argument(
        "--min_pitch",
        type=int,
        default=21,
        help="Minimum midi pitch that can be written in the image",
    )
    parser.add_argument(
        "--max_pitch",
        type=int,
        default=108,
        help="Maximum midi pitch that can be written in the image.",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="Music BPM",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if not args.save:
        port = mido.open_output('midi_port')

    for fname in args.files:
        img = cv.imread(fname)
        # img, time_factor, pixel_per_beat, min_pitch, max_pitch, bpm
        midi = img_to_midi(img, 24, args.pixel_per_beat, args.min_pitch, args.max_pitch, args.bpm)

        if args.save:
            midi.save(os.path.join(args.output_dir, os.path.splitext(os.path.basename(fname))[0] + "_converted.mid"))
        else:
            for msg in midi.Play():
                port.send(msg)

# ====


if __name__ == "__main__":
    main()