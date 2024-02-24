#!/usr/bin/python3

import sys
import os
import argparse
import logging

from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import numpy as np

import cv2 as cv


def main():
    parser = argparse.ArgumentParser(description="Create a dataset from midi files.")
    parser.add_argument("files", type=str, nargs="*", help="Input files")
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging level to INFO"
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=256,
        help="Image width, corresponding to the pitch",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=256,
        help="Image height, corresponding to the time",
    )
    parser.add_argument(
        "--pixel_per_beat",
        type=int,
        default=4,
        help="Number of pixel (vertical) in a beat",
    )
    parser.add_argument(
        "--overlap_count",
        type=int,
        default=1,
        help="Number of time a segment is represented in the dataset.",
    )
    # Default is piano range
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
        "--test_convert_back",
        action="store_true",
        help="Convert back some of the image to test that the conversion yields correct results",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # not included
    args.max_pitch = args.max_pitch + 1

    for fname in args.files:
        midi = MidiFile(fname)
        name = os.path.splitext(os.path.basename(fname))[0]

        time_factor = (
            midi.ticks_per_beat / args.pixel_per_beat
        )  # each pixel is time_factor long - ticks per pixel
        logging.info("file: %s - time_factor=%d" % (fname, time_factor))

        for i in range(args.overlap_count):
            offset = i * (args.img_width / args.overlap_count)
            midi_to_dataset(midi, name, offset, time_factor, args)


# ====


def midi_to_dataset(midi, name, offset, time_factor, args):
    # Only one track
    track0 = midi.tracks[0]
    midi_it = iter(track0)
    i = 0
    # skip offset
    msg = None
    msg_offset = 0

    if offset > 0:
        msg_offset = -time_factor * offset

        while msg_offset < 0:
            msg = next(midi_it)
            msg_offset += msg.dict()["time"]

    while True:
        img = np.zeros((args.img_height, args.img_width, 3), dtype=np.uint8)
        try:
            msg, msg_offset = draw_midi(
                midi_it,
                img,
                time_factor,
                args.min_pitch,
                args.max_pitch,
                msg,
                msg_offset,
            )
        except StopIteration:
            break
        finally:
            out_img_name = os.path.join(
                args.output_dir, name + "_offset%d_%d.png" % (offset, i)
            )
            cv.imwrite(out_img_name, img)
            logging.info("Wrote image file %s" % out_img_name)

            if args.test_convert_back and i == 0 and offset == 0:
                converted_midi = img_to_midi(
                    img,
                    time_factor,
                    args.pixel_per_beat,
                    args.min_pitch,
                    args.max_pitch,
                )
                out_midi_name = os.path.join(args.output_dir, name + "_converted.mid")
                converted_midi.save(out_midi_name)
                logging.info("Wrote midi file %s" % out_midi_name)

        i += 1


def draw_midi(
    midi_it, img, time_factor, min_pitch, max_pitch, last_msg=None, last_msg_offset=0
):
    max_time = img.shape[1] * time_factor
    pedal_size = img.shape[0] - (max_pitch - min_pitch) * 2

    if last_msg is not None:
        msg = last_msg
        # hack to get the correct time
        time = last_msg_offset - msg.dict()["time"]
    else:
        msg = next(midi_it)
        time = 0

    current_notes = dict()
    pedal_start = -1

    while True:
        d = msg.dict()
        time += d["time"]

        if time > max_time:
            offset = time - max_time
            return msg, offset

        if "note" in d:
            note = d["note"]

        if msg.is_cc(control=64):  # pedal
            if pedal_start >= 0 and d["value"] < 64:
                # stop pedaling and write pedal
                draw_pedal(
                    img, pedal_start / time_factor, time / time_factor, pedal_size
                )
                pedal_start = -1
            elif pedal_start < 0 and d["value"] >= 64:
                pedal_start = time

        elif msg.type == "note_on":
            # print(msg.type, type(d["note"]), d["velocity"], d["time"])
            current_notes[note] = {"time": time, "velocity": d["velocity"]}
        elif msg.type == "note_off":
            if note in current_notes:
                start = current_notes[note]
                draw_note(
                    img,
                    start["time"] / time_factor,
                    time / time_factor,
                    note,
                    start["velocity"],
                    min_pitch,
                    max_pitch,
                )
                del current_notes[note]

        msg = next(midi_it)


def draw_note(img, start, end, pitch, velocity, min_pitch, max_pitch):
    """
    img = OpenCV image
    start = which pixel to start on the image. float
    end = which pixel to end on the image. float
    pitch = midi pitch
    velocity = midi velocity
    min_pitch & max_pitch as argument of the program. Only max_pitch is used but we keep the two for versatile API
    """
    # print("draw_note %0.1f-%0.1f" % (start, end))
    start_x, start_off, end_x, end_off = get_x_and_off(start, end)

    start_off_val = np.uint8((start_off + 0.5) * 255)
    end_off_val = np.uint8((end_off + 1.5) / 2 * 255)
    velocity_val = np.uint8(velocity * 2)
    y = max_pitch - pitch
    img[y * 2 : (y + 1) * 2, start_x:end_x] = [end_off_val, start_off_val, velocity_val]


def draw_pedal(img, start, end, pedal_size):
    # print("draw_pedal %0.1f-%0.1f" % (start, end))
    start_x, start_off, end_x, end_off = get_x_and_off(start, end)
    start_off_val = np.uint8((start_off + 0.5) * 255)
    end_off_val = np.uint8((end_off + 1.5) / 2 * 255)
    img[img.shape[0] - pedal_size : img.shape[0], start_x:end_x] = [
        end_off_val,
        start_off_val,
        255,
    ]


def get_x_and_off(start, end):
    start_x = round(start)
    start_off = start - start_x
    end_x = round(end)
    end_off = end - end_x
    # edge case: short notes
    if start_x == end_x:
        end_x += 1
        # end_off lands on [-1.5, -0.5]
        end_off -= 1
    return start_x, start_off, end_x, end_off


def img_to_midi(img, time_factor, pixel_per_beat, min_pitch, max_pitch, bpm=120, tolerance=3):
    """
    tolerance: Normally, a single note is encoded as a set of 3 values that
    remain constant over the span of the note. This can be used to detect note
    changes. Since we're decoding machine produced image, we're note sure the
    value will remain constant, so this is a small tolerance to take the
    variations into account.
    """
    note_range = (max_pitch - min_pitch) * 2
    pedal_size = img.shape[0] - note_range
    # a pixel that has all its values under this threshold is considered black
    BLACK_THRESH = 32

    pedal_start_x = -1
    current_notes = dict()  # pitch: note data

    events = []

    for x in range(img.shape[1]):
        # notes
        notes_slice = img[0:note_range, x]
        for y in range(notes_slice.shape[0] // 2):
            # y axis is upside down on cv images
            pitch = max_pitch - y
            note_val = np.mean(notes_slice[y * 2 : (y + 1) * 2], axis=0)
            has_note = np.any(note_val > BLACK_THRESH)
            # print(note_val, has_note)

            if pitch in current_notes:
                last_note = current_notes[pitch]
                if (
                    not has_note
                    or np.max(np.abs(note_val - last_note["val"])) > tolerance
                ):
                    full_note = np.mean(
                        img[y * 2 : (y + 1) * 2, last_note["start_x"] : x], axis=(0, 1)
                    )
                    start_off_val = (full_note[1] / 255) - 0.5
                    end_off_val = (full_note[0] / 255) * 2 - 1.5
                    velocity = full_note[2] / 2
                    events.append(
                        {
                            "type": "note_on",
                            "note": pitch,
                            "velocity": int(velocity),
                            "time": round(
                                (last_note["start_x"] + start_off_val) * time_factor
                            ),
                        }
                    )
                    events.append(
                        {
                            "type": "note_off",
                            "note": pitch,
                            "velocity": int(velocity),
                            "time": round((x + end_off_val) * time_factor),
                        }
                    )
                    del current_notes[pitch]
                else:
                    # update val
                    last_note["val"] = note_val

            if pitch not in current_notes and has_note:
                current_notes[pitch] = {
                    "start_x": x,
                    "val": note_val,
                }

        # pedal
        pedal_slice = img[img.shape[0] - pedal_size : img.shape[0], x]
        pedaling = np.mean(pedal_slice, axis=0)[2] > 128

        if pedal_start_x < 0 and pedaling:
            pedal_start_x = x
        elif pedal_start_x >= 0 and not pedaling:
            full_pedal = np.mean(
                img[img.shape[0] - pedal_size : img.shape[0], pedal_start_x:x],
                axis=(0, 1),
            )
            start_off_val = (full_pedal[1] / 255) - 0.5
            end_off_val = (full_pedal[0] / 255) * 2 - 1.5
            events.append(
                {
                    "type": "control_change",
                    "value": 127,
                    "time": round((pedal_start_x + start_off_val) * time_factor),
                }
            )
            events.append(
                {
                    "type": "control_change",
                    "value": 0,
                    "time": round((x + end_off_val) * time_factor),
                }
            )
            pedal_start_x = -1

    events.sort(key=lambda x: x["time"])

    midi = MidiFile(type=1, ticks_per_beat=int(pixel_per_beat * time_factor))
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm), time=0))
    # for fun instrument change
    # track.append(Message("program_change", program=12, time=0))

    last_time = 0
    for evt in events:
        time = evt["time"] - last_time
        if evt["type"] == "control_change":
            track.append(
                Message(
                    "control_change",
                    channel=0,
                    control=64,
                    value=evt["value"],
                    time=time,
                )
            )
        elif evt["type"] in ["note_on", "note_off"]:
            track.append(
                Message(
                    evt["type"], note=evt["note"], velocity=evt["velocity"], time=time
                )
            )
        last_time = evt["time"]

    return midi


if __name__ == "__main__":
    main()
