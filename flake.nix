# Untested starter: adjust to taste. `nix develop` should give Python + uv.
{
  description = "bagua-weather dev shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.python312 pkgs.uv pkgs.git ];
          shellHook = ''
            [ -d .venv ] || uv venv
            . .venv/bin/activate
          '';
        };
      });
    };
}
