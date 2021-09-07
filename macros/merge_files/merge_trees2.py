import ROOT

#files = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/ana_alldaughter_files.txt"
#files = "/Users/jsen/tmp/pion_qe/2gev_full_mc_sample/newShower_n14580_no_alldaughter/2gev_full_mc.txt"
files = "/Users/jsen/tmp/pion_qe/2gev_full_mc_sample/newShower_n14580_no_alldaughter_all/all_files.txt"
tree_name = "pduneana/beamana"

with open(files) as f:
    file_list = f.readlines()

file_list = [line.strip() for line in file_list]

treeList = ROOT.TList()
outputFile = ROOT.TFile('MergeTest.root', 'recreate')
pyfilelist = []
pytreelist = []

for path in file_list:
        print("Path", path)
        inputFile = ROOT.TFile(path, 'read')
        pyfilelist.append(inputFile) # Make this TFile survive the loop!
        inputTree = inputFile.Get(tree_name)
        pytreelist.append(inputTree) # Make this TTree survive the loop!
        outputTree = inputTree.CloneTree() #instead of extensive processing
        treeList.Add(outputTree)

outputFile.cd()
outputTree = ROOT.TTree.MergeTrees(treeList)
outputFile.Write()
outputFile.Close()
