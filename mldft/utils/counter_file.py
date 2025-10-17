from pathlib import Path

counter_file_path = "/path/to/your/directory/.counter"


def get_and_increment_counter(counter_file_path: str) -> int:
    """A function that gets and increments a counter stored in a file. Problems will arise if
    multiple processes try to access the file at the same time.

    Args:
        counter_file_path (str): Path to the file where the counter is stored.

    Returns:
        int: The value of the counter after incrementation.
    """
    counter_file_path = Path(counter_file_path)
    if not counter_file_path.exists():
        # If the models directory is empty, we need to create the train and runs dir as well.
        if not counter_file_path.parent.exists():
            counter_file_path.parent.mkdir(parents=True)
        with open(counter_file_path, "w") as file:
            file.write("0")
        return 0
    else:
        with open(counter_file_path, "r+") as file:
            current_value = int(file.read().strip())
            new_value = current_value + 1
            file.seek(0)
            file.truncate()
            file.write(str(new_value))

        return new_value
