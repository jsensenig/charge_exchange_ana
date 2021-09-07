
void fit_hist( const std::string &name, TH1D* hist ) {

  // Scale the histogram integrals to 1
  hist->Scale( 1. / hist->Integral() );

  TCanvas c = TCanvas();
  gStyle->SetOptFit(0x2);

  // Fit beam start position or angle
  TF1 f("f", "gaus");
  hist->Fit("f");
  hist->Draw();

  // .. and save to a .pdf
  std::string pdf_name = "beam_" + name + "_fit.pdf";
  c.SaveAs(pdf_name.c_str());

}

void beam_quality() {

  std::string fname = "/Users/jsen/tmp/pion_qe/ana_scripts/merge_files/output.root";
  std::string treename = "beamana;17";

  TFile *file_address = TFile::Open(fname.c_str());

  if( !file_address -> IsOpen() ) {
    std::cout << "File " << fname << " not open!" << std::endl;
    return;
  }

  TTree *tree = (TTree*)file_address->Get(treename.c_str());

  double reco_beam_calo_startX, reco_beam_calo_startY, reco_beam_calo_startZ;
  double reco_beam_calo_endX, reco_beam_calo_endY, reco_beam_calo_endZ;

  tree->SetBranchAddress("reco_beam_calo_startX", &reco_beam_calo_startX);
  tree->SetBranchAddress("reco_beam_calo_startY", &reco_beam_calo_startY);
  tree->SetBranchAddress("reco_beam_calo_startZ", &reco_beam_calo_startZ);
  tree->SetBranchAddress("reco_beam_calo_endX", &reco_beam_calo_endX);
  tree->SetBranchAddress("reco_beam_calo_endY", &reco_beam_calo_endY);
  tree->SetBranchAddress("reco_beam_calo_endZ", &reco_beam_calo_endZ);

  std::map<std::string, TH1D*> hist_map;

  TH1D* hx = hist_map["startx"] = new TH1D("hx", "Beam Calo Start X;X [cm];Count", 50, -80., 20.);
  TH1D* hy = hist_map["starty"] = new TH1D("hy", "Beam Calo Start Y;Y [cm];Count", 40, 350., 500.);
  TH1D* hz = hist_map["startz"] = new TH1D("hz", "Beam Calo Start Z;Z [cm];Count", 25, -5., 10.);

  TH1D* thetax = hist_map["anglex"] = new TH1D("thetax", "Beam Angle #theta_x;#theta_x [deg];Count", 60, 0.0, 180.0);
  TH1D* thetay = hist_map["angley"] = new TH1D("thetay", "Beam Angle #theta_y;#theta_y [deg];Count", 60, 0.0, 180.0);
  TH1D* thetaz = hist_map["anglez"] = new TH1D("thetaz", "Beam Angle #theta_z;#theta_z [deg];Count", 30, 0.0, 90.0); 
  
  size_t nevts = tree -> GetEntries();

  for ( size_t evt = 0; evt < nevts; evt++ ) {
    tree->GetEntry( evt );

    // Fill the beam x,y,x
    hx->Fill( reco_beam_calo_startX );
    hy->Fill( reco_beam_calo_startY );
    hz->Fill( reco_beam_calo_startZ );

    // Beam angles
    TVector3 beam_start(reco_beam_calo_startX, reco_beam_calo_startY, reco_beam_calo_startZ);
    TVector3 beam_end(reco_beam_calo_endX, reco_beam_calo_endY, reco_beam_calo_endZ);
    TVector3 beam_dir = (beam_end - beam_start).Unit();

    // A large number of events have 0 angle, skip them so they don't skew the fit.
    // These will be rejected by the event selection anyway
    if( beam_dir.X() == 0.0 ) continue;

    // theta_i = arccos( i * cos(theta_i) )  where i = {x,y,z}
    // practically: theta_i = arccos( beam_dir.i() ) where  i={x,y,z} and beam_dir normalized
    thetax->Fill( TMath::RadToDeg() * TMath::ACos(beam_dir.X()) );
    thetay->Fill( TMath::RadToDeg() * TMath::ACos(beam_dir.Y()) );
    thetaz->Fill( TMath::RadToDeg() * TMath::ACos(beam_dir.Z()) );

  }

  // Normalize, fit and save the histograms
  for( auto &map : hist_map ) fit_hist( map.first, map.second );

}
