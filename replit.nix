
{ pkgs }: {
  deps = [
    pkgs.python39
    pkgs.python39Packages.flask
    pkgs.python39Packages.joblib
    pkgs.python39Packages.numpy
    pkgs.python39Packages.scikit-learn
  ];
}
