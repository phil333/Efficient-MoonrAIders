import rosbag

import itertools
import operator

def step_and_align_generators(images, labels):
    ''' Aligns both generator based on matching timestamp '''
    _, image, _ = next(images)
    _, label, _ = next(labels)

    if image.header.stamp < label.header.stamp:
        while image.header.stamp != label.header.stamp:
            _, image, _ = next(images)
    elif image.header.stamp > label.header.stamp:
        while image.header.stamp != label.header.stamp:
            _, label, _ = next(labels)

    if image.header.stamp != label.header.stamp:
        raise ValueError("Can't align img and label topics!")

    return image, label

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def data_generator(
    path_to_rosbags,
    limit_by_N=None,
    source_robot_name=None,
    skip_messages_without_labels=True,
    batches=None,
    yield_filename=False
    ):
    """
        Generates (image, label) pairs from a source directory containing rosbags.
        Optionally generates (path_to_rosbag, image, label) tuples if
        yield_filename argument is true

        Parameters
        ----------
        path_to_rosbags : pathlib.Path
            Path to directory which contains rosbags
        limit_by_N : int, optional
            Generate only N pairs
        source_robot_name : str, optional
            Name of the robot that was used to generate data, by default it is
            'small_scout_1'
        skip_messages_without_labels : bool, optional
            If true, then messages which do not contain any labels
            (i.e. there were no objects in the field of view)
            will not be yieled by the generator
        batches : int, optional
            If a number N is given, then the resulting generator will yield
            batches of N size. If None is passed in (default), the generator
            will yield pair by pair. Cannot be used in conjunction with
            yield_filename
        yield_filename : bool, optional
            If true, then the resulting iterator will also yield the filename
            of the Rosbag that produced that pair. Cannot be used together with
            batches option.

        Returns
        -------
        Iterator
            an iterator that yields pairs (image, label), where image is a
            sensor_msgs.msg.Image and label is label_msgs.msg.ImageLabel
    """

    if source_robot_name is None:
        raise ValueError('source_robot_name cannot be None!')

    if yield_filename and isinstance(batches, int):
        raise ValueError('yield_filename and batches cannot be used together!')

    def generator():

        for path in path_to_rosbags.glob('**/*.bag'):

            with rosbag.Bag(path) as bag:

                images = bag.read_messages(topics=[f'/{source_robot_name}/camera/left/image_raw_repub'])
                labels = bag.read_messages(topics=[f'/{source_robot_name}/labels'])

                while True:
                    try:
                        image, label = step_and_align_generators(images, labels)

                        if skip_messages_without_labels:
                            while len(label.object_labels) == 0:
                                image, label = step_and_align_generators(images, labels)

                        if yield_filename:
                            yield path, image, label
                        else:
                            yield image, label

                    except StopIteration:
                      break

    gen = generator()

    if isinstance(limit_by_N, int):
        gen = itertools.islice(gen, limit_by_N)

    if isinstance(batches, int):
        img_gen, label_gen = itertools.tee(gen)
        img_gen = map(operator.itemgetter(0), img_gen)
        label_gen = map(operator.itemgetter(1), label_gen)
        img_gen = grouper(batches, img_gen)
        label_gen = grouper(batches, label_gen)
        gen = zip(img_gen, label_gen)

    yield from gen
