import cv2
import glob
import os
import shutil

# Define old and new base paths
OLD_BASE = "/Documents/VendorMobile"
NEW_BASE = "/Documents/vendor_triplet"
SEQ_PATH = os.path.join(NEW_BASE, "sequences")

SEQ_LIST = {}
REASSIGN_LIST = {
    "12-13-14-20-02": "00001",  # genshinimpact_2nd-veryhigh_taaon_60fps-battle_pvp
    "12-13-14-29-10": "00002",  # genshinimpact_2nd-veryhigh_taaon_60fps-battle_wild
    "12-13-13-59-49": "00003",  # genshinimpact_2nd-veryhigh_taaon_60fps-chat_character
    "12-13-14-07-26": "00004",  # genshinimpact_2nd-veryhigh_taaon_60fps-indoor_move
    "12-13-14-25-16": "00005",  # genshinimpact_2nd-veryhigh_taaon_60fps-interface_role
    "12-13-13-59-14": "00006",  # genshinimpact_2nd-veryhigh_taaon_60fps-map_town
    "12-13-14-27-54": "00007",  # genshinimpact_2nd-veryhigh_taaon_60fps-map_wild
    "11.03_10.24.52": "00008",  # genshinimpact_rdc-veryhigh_taaon_30fps-battle_pvp
    "11.03_10.38.54": "00009",  # genshinimpact_rdc-veryhigh_taaon_30fps-chat_character
    "11.02_16.56.18": "00010",  # genshinimpact_rdc-veryhigh_taaon_30fps-indoor_move
    "11.03_10.57.59": "00011",  # genshinimpact_rdc-veryhigh_taaon_30fps-interface_role
    "11.02_16.17.56": "00012",  # genshinimpact_rdc-veryhigh_taaon_30fps-map_town
    "11.02_15.54.46": "00013",  # genshinimpact_rdc-veryhigh_taaon_30fps-map_wild
    "11.03_16.13.16": "00014",  # pubg-low_90fps-indoor_move
    "11.03_15.48.25": "00015",  # pubg-low_90fps-len_close
    "11.22_15.56.23": "00016",  # pubg-low_90fps-len_open
    "11.03_16.09.42": "00017",  # pubg-low_90fps-map_parachute
    "11.03_16.21.48": "00018",  # pubg-low_90fps-map_run
    "11.03_14.21.29": "00019",  # pubg-veryhigh_lowfps-indoor_move
    "11.03_16.16.41": "00020",  # pubg-veryhigh_lowfps-len_close
    "11.03_16.15.58": "00021",  # pubg-veryhigh_lowfps-len_open
    "11.03_15.09.47": "00022",  # pubg-veryhigh_lowfps-map_car
    "11.03_14.09.13": "00023",  # pubg-veryhigh_lowfps-map_parachute
    "11.02_15.28.23": "00024",  # wzry-veryhigh_extremehighfps-battle_multi_hero
    "11.01_15.53.07": "00025",  # wzry-veryhigh_extremehighfps-battle_single_hero
    "11.01_15.41.48": "00026",  # wzry-veryhigh_extremehighfps-Interface_role
    "11.02_16.07.01": "00027",  # wzry-veryhigh_extremehighfps-mini_game
}


def new_sequence_id(file_path):
    seq_id = "00000"
    for old_path, new_path in REASSIGN_LIST.items():
        if old_path in file_path:
            if seq_id == "00000":
                seq_id = new_path
            else:
                raise Exception(
                    "Error! Check REASSIGN_LIST - Found duplicate keys: {}".format(
                        old_path
                    )
                )
    if seq_id == "00000":
        raise Exception("Error! Key not found: {}".format(file_path))
    return seq_id


def process_image(hr_path, lower_quality=False):
    hr_img = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
    height, width = hr_img.shape[:2]

    if lower_quality:
        # Resize to half of original size and resize back
        hr_img = cv2.resize(
            hr_img, (width // 2, height // 2), interpolation=cv2.INTER_AREA
        )
        hr_img = cv2.resize(hr_img, (width, height), interpolation=cv2.INTER_AREA)

    # Convert to RGBA if not already
    if hr_img.shape[2] == 3:
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_RGB2RGBA)

    if hr_img.shape[2] != 4:
        raise Exception("Error! Check image depth: {}".format(hr_path))
    return hr_img


if __name__ == "__main__":
    # Ensure the new structure exists
    os.makedirs(NEW_BASE, exist_ok=False)
    os.makedirs(SEQ_PATH, exist_ok=False)
    print("Create dataset directory:", NEW_BASE)
    print("Sequences will be saved to:", SEQ_PATH)

    png_files = glob.glob(f"{OLD_BASE}/**/*.png", recursive=True)
    png_files.sort()

    current_id = "00000"
    seq_folder = ""
    frame_set = 1
    for i in range(len(png_files) - 2):
        seq_folder_1 = new_sequence_id(png_files[i])
        seq_folder_3 = new_sequence_id(png_files[i + 2])
        if seq_folder_1 != seq_folder_3:  # Scene change occurs
            continue
        if seq_folder_1 != current_id:
            seq_folder = os.path.join(SEQ_PATH, seq_folder_1)
            os.makedirs(seq_folder, exist_ok=False)

            current_id = seq_folder_1
            frame_set = 0
            print("Processing new sequence:", seq_folder)
        if seq_folder == "":
            raise Exception("Error: Sequence id not assigned")

        # Image set
        # im1.png -> lr input frame
        # im2.png -> hr gt interp
        # im3.png -> lr input frame
        set_folder = os.path.join(seq_folder, str(frame_set).zfill(4))
        os.makedirs(set_folder, exist_ok=False)
        cv2.imwrite(
            os.path.join(set_folder, "im1.png"),
            process_image(png_files[i], lower_quality=True),
        )
        cv2.imwrite(
            os.path.join(set_folder, "im2.png"),
            process_image(png_files[i + 1], lower_quality=False),
        )
        cv2.imwrite(
            os.path.join(set_folder, "im3.png"),
            process_image(png_files[i + 2], lower_quality=True),
        )
        frame_set += 1
