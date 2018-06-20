import fire

def remove_the_same_near_row(read_file):
    # notice: if use xreadlines will overlap the read_file
    freads = open(read_file, 'r').readlines();
    fwrite = open(read_file, 'w');
    preline = "\n";
    for line in freads:
        if(line == preline):
            continue;
        fwrite.write(line);
        preline = line;




if __name__ == "__main__":
    fire.Fire(remove_the_same_near_row);
