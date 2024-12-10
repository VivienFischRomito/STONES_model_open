# -*- coding: utf-8 -*-
import os
    
def creat_path(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError:
        print("Creating output path %s failed" % path)
    else:
        pass
        #print("The directory %s checked" % path)


def main(name):
    ### Input and output path
    input_path = './inputs/' + name
    output_path = './outputs'
    plots_path = output_path + '/plots'
    
    creat_path(output_path)
    creat_path(output_path + '/Scenario')
    creat_path(plots_path)
    
    return input_path, output_path, plots_path

#If executed in the main program, execute the code
#if __name__ == '__main__':
#    main(name)