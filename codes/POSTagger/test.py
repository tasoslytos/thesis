import subprocess

subprocess.call('stanford-postagger models\wsj-0-18-left3words-distsim.tagger sample-input.txt > my_output.txt', shell=True)