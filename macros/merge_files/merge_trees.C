
std::vector<std::string> LoadFileList( const std::string& file_list ) {

  std::vector<std::string> file_vec;
  std::string line;

  std::ifstream file( file_list );

  if ( file.is_open()) {
    while ( getline( file, line )) {
      std::cout << "Loading file: " << line << std::endl;
      file_vec.emplace_back( line );
    }
    file.close();
  } else std::cout << "Unable to open file " << file_list << std::endl;

  return file_vec;

}

void merge_trees() {

  std::string files = "/Users/jsen/tmp/pion_qe/2gev_full_mc_sample/newShower_n14580_no_alldaughter_all/all_files.txt";
  std::vector<std::string> flist = LoadFileList(files);
  TList *list = new TList;
  
  ///Loop on all .root output files
  for (auto &f : flist) {
    TFile *file_adress = TFile::Open(f.c_str());
    TTree *tree_adress = (TTree*)file_adress->Get("pduneana/beamana");
    list->Add(tree_adress);
  }
  
  // Output tree:
  TFile* outputfile = TFile::Open("output.root", "recreate");
  TTree *TotalTree = TTree::MergeTrees(list);

  TotalTree->Write();
  outputfile->Close();
  delete outputfile;

}
