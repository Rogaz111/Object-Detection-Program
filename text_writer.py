import datetime

now = datetime.datetime.now()
unique_entries = {}

def write_to_file(detection_input):
    global unique_entries
    if detection_input and detection_input not in unique_entries:
        unique_entries[detection_input] = str(now)
        with open('detect.text.txt', 'a') as f:
            f.write(f"{unique_entries[detection_input]}: {detection_input}\n")