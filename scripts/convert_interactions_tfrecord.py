"""
Convert MicroLens-100K interaction CSV to TFRecord format.
Uses leave-one-out split: last item = test, second-to-last = val, rest = train.

Usage:
    python scripts/convert_interactions_tfrecord.py \
        --input data/microlens/MicroLens-100k_pairs.csv \
        --output data/microlens/tfrecords/
"""

import argparse
import os
import tensorflow as tf
import pandas as pd
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to interactions CSV")
    parser.add_argument("--output", required=True, help="Output directory for TFRecords")
    return parser.parse_args()


def write_tfrecord(sequences, filepath):
    with tf.io.TFRecordWriter(filepath) as writer:
        for user_id, item_ids in sequences.items():
            feature = {
                "user_id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[user_id])
                ),
                "item_ids": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=item_ids)
                ),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input)
    df.columns = ["user_id", "item_id", "timestamp"]
    df = df.sort_values(["user_id", "timestamp"])

    user_sequences = defaultdict(list)
    for _, row in df.iterrows():
        user_sequences[int(row["user_id"])].append(int(row["item_id"]))

    train_seqs, val_seqs, test_seqs = {}, {}, {}
    for user_id, items in user_sequences.items():
        if len(items) < 3:
            continue
        train_seqs[user_id] = items[:-2]
        val_seqs[user_id] = items[:-1]
        test_seqs[user_id] = items

    write_tfrecord(train_seqs, os.path.join(args.output, "train.tfrecord"))
    write_tfrecord(val_seqs, os.path.join(args.output, "val.tfrecord"))
    write_tfrecord(test_seqs, os.path.join(args.output, "test.tfrecord"))

    print(f"Done! Train: {len(train_seqs)} users, Val: {len(val_seqs)}, Test: {len(test_seqs)}")


if __name__ == "__main__":
    main()
