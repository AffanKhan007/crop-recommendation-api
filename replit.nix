{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.python311Packages.pip
    pkgs.python311Packages.setuptools
    pkgs.python311Packages.flask
    pkgs.python311Packages.joblib
  ];

  postInstall = ''
    pip install numpy==1.26.4
    pip install scikit-learn==1.6.1
  '';
}
