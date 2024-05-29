
std::vector<std::string> LoadFileList( const std::string& file_list ) {

  std::vector<std::string> file_vec;
  std::string line;
  int nfiles = 0;

  std::ifstream file( file_list );

  if ( file.is_open()) {
    while ( getline( file, line )) {
      file_vec.emplace_back( line );
      nfiles++;
    }
    file.close();
  } else std::cout << "Unable to open file " << file_list << std::endl;

  std::cout << "Loaded " << nfiles << " files" << std::endl;

  return file_vec;

}

double MakeIncidentEnergies(std::vector<double> *true_beam_traj_Z,
                              std::vector<double> *true_beam_traj_KE,
                              std::vector<double> *true_beam_new_incidentEnergies) {

  // Only include trajectory points starting in the active volume
  double fTrajZStart = -0.49375;
  // Only include trajectory points less than slice 464 (the end of APA3)
  int fSliceCut = 464;
  // ProtoDUNE TPC wire pitch [cm]
  double fPitch = 0.4794; 

  double true_beam_new_interactingEnergy = -999;
  double next_slice_z = fTrajZStart;
  int next_slice_num = 0;

  for (size_t j = 1; j < true_beam_traj_Z->size() - 1; ++j) {
    double z = true_beam_traj_Z->at(j);
    double ke = true_beam_traj_KE->at(j);

    if (z < fTrajZStart) continue;

    if (z >= next_slice_z) {
      double temp_z = true_beam_traj_Z->at(j-1);
      double temp_e = true_beam_traj_KE->at(j-1);

      while (next_slice_z < z && next_slice_num < fSliceCut) {
        double sub_z = next_slice_z - temp_z;
        double delta_e = true_beam_traj_KE->at(j-1) - ke;
        double delta_z = z - true_beam_traj_Z->at(j-1);
        temp_e -= (sub_z/delta_z)*delta_e;
        true_beam_new_incidentEnergies->push_back(temp_e);
        temp_z = next_slice_z;
        next_slice_z += fPitch;
        ++next_slice_num;
      }
    }
  }
  // If the trajectory does not reach the end of the fiducial slices it must have interacted.
  // The interacting energy will be the last incident energy.
  if( next_slice_num < fSliceCut && true_beam_new_incidentEnergies->size() > 0 ) {
    true_beam_new_interactingEnergy = true_beam_new_incidentEnergies->back();
  }

  return true_beam_new_interactingEnergy;
}

//truncated mean of SIGMA = cutting %
void truncatedMean( std::vector<std::vector<double>> *vecs_dEdX, std::vector<double> *dEdX_truncated_mean ) {

    // Upper and lower 16%
   double truncate_low = 0.16;
   double truncate_high = 0.16;
   size_t size = 0;
   std::vector<double> help_vec;
   truncate_high = 1 - truncate_high;
   int i_low = 0;
   int i_high = 0;

   //sort the dEdX vecotrs in matrix
   for( auto &vec : *vecs_dEdX ) {
      size = vec.size();
      help_vec.clear();

      //check dEdX vector isn't empty!
      if( vec.empty() ) {
         dEdX_truncated_mean->push_back(-999.);
         continue;
      }

      else {
         // Sort Vector
         std::sort(vec.begin(), vec.end());

         //Discard upper and lower part of signal
         i_low = rint( size*truncate_low );
         i_high = rint( size*truncate_high );


         for(int i = i_low; i <= i_high; i++) help_vec.push_back(vec[i]);

         //Mean of help vector
         dEdX_truncated_mean->push_back(accumulate(help_vec.begin(), help_vec.end(), 0.0) / help_vec.size());
      }
   }
}

void GetRecoTrackLength( std::vector<double> *reco_beam_calo_X, std::vector<double> *reco_beam_calo_Y, std::vector<double> *reco_beam_calo_Z, std::vector<double> *reco_track_cumlen ) {

  double reco_trklen = -999;

  for ( size_t i = 1; i < reco_beam_calo_Z->size(); i++ ) {
    if (i == 1) reco_trklen = 0;
    reco_trklen += sqrt( pow( (*reco_beam_calo_X)[i] - (*reco_beam_calo_X)[i-1], 2)
                        + pow( (*reco_beam_calo_Y)[i] - (*reco_beam_calo_Y)[i-1], 2)
                        + pow( (*reco_beam_calo_Z)[i] - (*reco_beam_calo_Z)[i-1], 2));
    reco_track_cumlen->push_back(reco_trklen);
  }

}
////////////////////////////////

void AddIncidentEnergyBranch(TFile * file) {

    TTree *tree = (TTree*)file->Get("pduneana/beamana");

    std::vector<std::vector<double>> *reco_daughter_allTrack_calibrated_dEdX_SCE = new std::vector<std::vector<double>>;
    std::vector<double> *dEdX_truncated_mean = new std::vector<double>;
    std::vector<double> *true_beam_traj_Z = new std::vector<double>;
    std::vector<double> *true_beam_traj_KE = new std::vector<double>;
    std::vector<double> *true_beam_traj_incidentEnergies = new std::vector<double>;
    std::vector<double> *reco_beam_calo_X = new std::vector<double>;
    std::vector<double> *reco_beam_calo_Y = new std::vector<double>;
    std::vector<double> *reco_beam_calo_Z = new std::vector<double>;
    std::vector<double> *reco_track_cumlen = new std::vector<double>;
    double true_beam_traj_interacting_Energy;

    TBranch *new_inc_energy = tree->Branch("true_beam_traj_incidentEnergies", &true_beam_traj_incidentEnergies);
    TBranch *new_int_energy = tree->Branch("true_beam_traj_interacting_Energy", &true_beam_traj_interacting_Energy);
    TBranch *new_truncated_mean = tree->Branch("dEdX_truncated_mean", &dEdX_truncated_mean);
    TBranch *new_reco_track_cumlen = tree->Branch("reco_track_cumlen", &reco_track_cumlen);
    tree->SetBranchAddress("true_beam_traj_Z", &true_beam_traj_Z);
    tree->SetBranchAddress("true_beam_traj_KE", &true_beam_traj_KE);
    tree->SetBranchAddress("reco_daughter_allTrack_calibrated_dEdX_SCE", &reco_daughter_allTrack_calibrated_dEdX_SCE);
    tree->SetBranchAddress("reco_beam_calo_X", &reco_beam_calo_X);
    tree->SetBranchAddress("reco_beam_calo_Y", &reco_beam_calo_Y);
    tree->SetBranchAddress("reco_beam_calo_Z", &reco_beam_calo_Z);

    size_t nentries = tree->GetEntries();
    for( size_t i = 0; i < nentries; i++ ) {
        tree->GetEntry(i);

        true_beam_traj_interacting_Energy = MakeIncidentEnergies(true_beam_traj_Z, true_beam_traj_KE, true_beam_traj_incidentEnergies);
        truncatedMean(reco_daughter_allTrack_calibrated_dEdX_SCE, dEdX_truncated_mean);

        GetRecoTrackLength( reco_beam_calo_X, reco_beam_calo_Y, reco_beam_calo_Z, reco_track_cumlen );

        new_inc_energy->Fill();
        new_int_energy->Fill();
        new_truncated_mean->Fill();
        new_reco_track_cumlen->Fill();

        dEdX_truncated_mean->clear();
        true_beam_traj_incidentEnergies->clear();
        reco_track_cumlen->clear();
    }

    tree->Write();

    delete dEdX_truncated_mean;
    delete true_beam_traj_Z;
    delete true_beam_traj_KE;
    delete true_beam_traj_incidentEnergies;
    delete reco_beam_calo_X;
    delete reco_beam_calo_Y;
    delete reco_beam_calo_Z;
    delete reco_track_cumlen;
}

void merge_trees() {

  std::string files = "/Users/jsen/work/Protodune/analysis/cex_event_selection/macros/merge_files/files.txt";
  std::vector<std::string> flist = LoadFileList(files);
  TList *list = new TList;

  // Maybe the new branch could be added in the Merge loop but I couldn't get it to
  // work in my brief attempt so do it here in a separate loop.
  for (auto &f : flist) {
    std::cout << "Adding branch to file: " << f << std::endl;
    TFile *file_adress = new TFile(f.c_str(), "update");
    AddIncidentEnergyBranch(file_adress);
    file_adress->Close();
    delete file_adress;
  }

  ///Loop on all .root output files
  for (auto &f : flist) {
    std::cout << "Adding file: " << f << std::endl;
    TFile *file_adress = TFile::Open(f.c_str());
    TTree *tree_adress = (TTree*)file_adress->Get("pduneana/beamana");
    list->Add(tree_adress);
  }
  
  std::cout << "Merging files! " << std::endl;

  // Output tree:
  TFile* outputfile = TFile::Open("output.root", "recreate");
  TTree *TotalTree = TTree::MergeTrees(list);

  TotalTree->Write();
  outputfile->Close();
  delete outputfile;

}
