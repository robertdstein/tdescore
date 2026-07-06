"""
API for interacting with the Babamul service.
"""

import os

from confluent_kafka import OFFSET_BEGINNING, Consumer, TopicPartition
from dotenv import load_dotenv

load_dotenv()


def get_conf(use_commit: bool = True) -> dict:
    """
    Get the Kafka configuration for Babamul.
    :param use_commit: Whether to enable auto commit
    :return: Kafka configuration dictionary
    """
    conf = {
        "bootstrap.servers": "kaboom.caltech.edu:9093",
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": os.getenv("BABAMUL_KAFKA_USERNAME"),
        "sasl.password": os.getenv("BABAMUL_KAFKA_PASSWORD"),
        "group.id": os.getenv("BABAMUL_KAFKA_GROUP"),
        "auto.offset.reset": "earliest",
    }
    # auto_conf = {
    #     "enable.auto.commit": True,
    # } if use_commit else {
    #     "enable.auto.commit": False,
    # }
    auto_conf = {"enable.auto.commit": use_commit}
    conf.update(auto_conf)
    return conf


def get_topic_message_count(topic_name: str) -> int:
    """
    Calculates the total number of messages in a Kafka topic.

    :param topic_name: Name of the Kafka topic
    :return: Total number of messages in the topic
    """
    consumer = Consumer(get_conf())

    # 1. Get metadata to find partition IDs
    metadata = consumer.list_topics(topic_name, timeout=10)
    if topic_name not in metadata.topics:
        return 0

    # Extract keys (partition IDs) from the partitions dictionary
    partition_ids = metadata.topics[topic_name].partitions.keys()

    total_count = 0
    for p_id in partition_ids:
        # 2. Create a TopicPartition object for querying
        tp = TopicPartition(topic_name, p_id)

        # 3. Query the broker for (low, high) offsets
        # This returns a tuple: (earliest_offset, latest_offset)
        low, high = consumer.get_watermark_offsets(tp, timeout=10)

        # Total messages in this partition = High Watermark - Low Watermark
        total_count += high - low

    consumer.close()
    return total_count


# def reset_topic_offsets(consumer: Consumer, topic_name: str) -> None:
#     """
#     Reset the offsets for all partitions of a Kafka topic to the beginning.
#
#     :param consumer: Kafka consumer instance
#     :param topic_name: Name of the Kafka topic
#     :return: None
#     """
#     # Get metadata to find partition IDs
#     metadata = consumer.list_topics(topic_name, timeout=10)
#     if topic_name not in metadata.topics:
#         return
#
#     partition_ids = metadata.topics[topic_name].partitions.keys()
#
#     # Create TopicPartition objects for all partitions
#     tps = [TopicPartition(topic_name, p_id, 0) for p_id in partition_ids]
#
#     # Assign the consumer to these partitions
#     consumer.assign(tps)
#
#     # Seek to the beginning of each partition
#     consumer.seek_to_beginning()
