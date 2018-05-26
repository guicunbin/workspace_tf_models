
import time


def process_one_line(line, boxes_li):
    assert len(boxes_li) == 0;
    line_li = line.strip().split(',');
    if(len(line_li)==0): return None;

    if(not line_li[0].endswith('.jpg')):
        return None;

    if(len(line_li) == 1 or len(line_li[1]) == 0):
        return line_li[0];

    print line_li
    nums_li = line_li[1].split(';');
    #print nums_li
    for nums in nums_li:
        boxes_li.append([int(nu) for nu in nums.split('_')])
    return line_li[0];


