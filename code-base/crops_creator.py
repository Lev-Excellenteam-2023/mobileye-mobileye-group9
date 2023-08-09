import json
from typing import Dict, Any

from PIL import Image
from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH, JSON_PATH, GRN, RED, IMAG_PATH

from pandas import DataFrame

def make_crop(*args, **kwargs):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    height_image_size=112
    width_image_size=44
    x=args[0]
    y=args[1]
    c=args[2]
    index=args[3]
    gtim_path=args[4]
    radius = args[5]
    height=radius*6
    width=radius*2
    side_h=0.1*height
    side_w=0.1*width
    x0=x+radius+side_w
    x1 = x - radius-side_w
    if c == RED:
        y1 = y + 5*radius+3*side_h
        y0= y - radius-side_h
    if c == GRN:
        y1 = y +radius+side_h
        y0 = y - 5*radius-3*side_h

    # Load the image from the specified path using the index
    image = Image.open(gtim_path)

    # Crop the image based on the specified coordinates
    cropped_image = image.crop((x1, y0, x0, y1))  # Note the order of coordinates
    # Resize the cropped image using a valid resampling filter (e.g., BILINEAR)
    resampled_image = cropped_image.resize((width_image_size, height_image_size), Image.BILINEAR)

    # Save the resized image to a file
    crop_path: str = 'C:/Users/user/Desktop/part1/mobileye-part_1/code/TFL_Detection_Pre/data/crops/my_crop_unique_name' + str(
        index) + '.png'
    resampled_image.save(CROP_DIR / crop_path)

    return x0, x1, y0, y1, cropped_image

def check_crop(*args, **kwargs):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """

    filename=args[6]
    x0 = args[2]
    x1 = args[3]
    y0 = args[4]
    y1 = args[5]
    with open(filename) as json_file:
        data = json.load(json_file)
    for i in range(len(data["objects"])):
        if data["objects"][i]["label"] == "traffic light":
            list_point = data["objects"][i]["polygon"]
            min_x = min(point[0] for point in list_point)
            max_x = max(point[0] for point in list_point)
            min_y = min(point[1] for point in list_point)
            max_y = max(point[1] for point in list_point)
            if x0 <= min_x and x1 >= max_x:
                if y0 <= min_y and y1 >= max_y:
                    return True, False
    return False, False


def create_crops(df: DataFrame) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        # example code:
        # ******* rewrite ONLY FROM HERE *******
        x0, x1, y0, y1, crop = make_crop(df[X][index], df[Y][index],df[COLOR][index],index,df[IMAG_PATH][index],df[RADIUS][index])

        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        crop_path: str = '/data/crops/my_crop_unique_name.probably_containing_the original_image_name+somthing_unique'
        # crop.save(CROP_DIR / crop_path)
        result_template[CROP_PATH] = crop_path
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(df[GTIM_PATH][index],
                                                                      crop, x0, x1, y0, y1,df[JSON_PATH][index],
                                                                      'everything_else_you_need_here')
        # ******* TO HERE *******

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)
    return result_df
