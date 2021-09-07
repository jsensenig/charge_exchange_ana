
void beam_quality() {

  std::string fname = "/Users/jsen/tmp/pion_qe/ana_scripts/merge_files/output.root";

  TFile *file_address = TFile::Open(fname.c_str());

  if( !file_address -> IsOpen() ) {
    std::cout << "File " << fname << " not open!" << std::endl;
    return;
  }

  TTree *tree = (TTree*)file_address->Get("beamana;17");


  double reco_beam_calo_startX, reco_beam_calo_startY, reco_beam_calo_startZ;
  double reco_beam_calo_endX, reco_beam_calo_endY, reco_beam_calo_endZ;

  tree->SetBranchAddress("reco_beam_calo_startX", &reco_beam_calo_startX);
  tree->SetBranchAddress("reco_beam_calo_startY", &reco_beam_calo_startY);
  tree->SetBranchAddress("reco_beam_calo_startZ", &reco_beam_calo_startZ);
  tree->SetBranchAddress("reco_beam_calo_endX", &reco_beam_calo_endX);
  tree->SetBranchAddress("reco_beam_calo_endY", &reco_beam_calo_endY);
  tree->SetBranchAddress("reco_beam_calo_endZ", &reco_beam_calo_endZ);

  TH1D hx("hx", "Beam Calo Start X;X [cm];Count", 50, -80., 20.);
  TH1D hy("hy", "Beam Calo Start Y;Y [cm];Count", 40, 350., 500.);
  TH1D hz("hz", "Beam Calo Start Z;Z [cm];Count", 25, -5., 10.);

  TH1D thetax("thetax", "Beam Angle #theta_x;#theta_x [deg];Count", 60, 0.0, 180.0);
  TH1D thetay("thetay", "Beam Angle #theta_y;#theta_y [deg];Count", 60, 0.0, 180.0);
  TH1D thetaz("thetaz", "Beam Angle #theta_z;#theta_z [deg];Count", 30, 0.0, 90.0);

  size_t nevts = tree -> GetEntries();

  for ( size_t evt = 0; evt < nevts; evt++ ) {
    tree->GetEntry( evt );

    // Fill the beam x,y,x
    hx.Fill( reco_beam_calo_startX );
    hy.Fill( reco_beam_calo_startY );
    hz.Fill( reco_beam_calo_startZ );

    // Beam angles
    TVector3 beam_start(reco_beam_calo_startX, reco_beam_calo_startY, reco_beam_calo_startZ);
    TVector3 beam_end(reco_beam_calo_endX, reco_beam_calo_endY, reco_beam_calo_endZ);
    TVector3 beam_dir = (beam_end - beam_start).Unit();

    // A large number of events have 0 angle, skip them so they don't skew the fit.
    // These will be rejected by the event selection anyway
    if( beam_dir.X() == 0.0 ) continue;

    // theta_i = arccos( i * cos(theta_i) )  where i = {x,y,z}
    // practically: theta_i = arccos( beam_dir.i() ) where  i={x,y,z} and beam_dir normalized
    thetax.Fill( TMath::RadToDeg() * TMath::ACos(beam_dir.X()) );
    thetay.Fill( TMath::RadToDeg() * TMath::ACos(beam_dir.Y()) );
    thetaz.Fill( TMath::RadToDeg() * TMath::ACos(beam_dir.Z()) );

  }

  // Scale the histogram integrals to 1
  hx.Scale( 1. / hx.Integral() );
  hy.Scale( 1. / hy.Integral() );
  hz.Scale( 1. / hz.Integral() );

  thetax.Scale( 1. / thetax.Integral() );
  thetay.Scale( 1. / thetay.Integral() );
  thetaz.Scale( 1. / thetaz.Integral() );


  TCanvas c = TCanvas();
  gStyle->SetOptFit(0x2);

  // Fit beam start position
  TF1 fx("fx", "gaus", -80., 20.);
  hx.Fit("fx", "R");
  hx.Draw();
  c.SaveAs("beam_start_x_fit.pdf");

  TF1 fy("fy", "gaus", 350., 500.);
  hy.Fit("fy", "R");
  hy.Draw();
  c.SaveAs("beam_start_y_fit.pdf");

  TF1 fz("fz", "gaus", -5., 10.);
  hz.Fit("fz", "R");
  hz.Draw();
  c.SaveAs("beam_start_z_fit.pdf");

  // Fit beam direction x,y,z angle
  TF1 fthetax("fthetax", "gaus", 0., 180.);
  thetax.Fit("fthetax", "R");
  thetax.Draw();
  c.SaveAs("beam_theta_x_fit.pdf");
                                    
  TF1 fthetay("fthetay", "gaus", 0., 180.);
  thetay.Fit("fthetay", "R");
  thetay.Draw();
  c.SaveAs("beam_theta_y_fit.pdf");
                                    
  TF1 fthetaz("fthetaz", "gaus", 0., 90.);
  thetaz.Fit("fthetaz", "R");
  thetaz.Draw();
  c.SaveAs("beam_theta_z_fit.pdf");

}
