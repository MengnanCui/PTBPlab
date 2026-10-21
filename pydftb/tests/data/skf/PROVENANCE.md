# Test parameter sets: provenance

All sets are stored gzip-compressed (`*.skf.gz`); pydftb reads them directly.
DFTB+ needs plain files: `pydftb.validation` decompresses them to a temporary
directory before running it. The sha256 sums below are of the **uncompressed** files.

## `ptbp_hcocu/` — H, C, O, Cu (Cu with an s, p, d basis)

Generated with this repository's own SKF generator:

```bash
ptbp gen H C O Cu --params PTBP
```

- Repository commit: `2eb6bd626542a89b5b115659d0c6861bb77ae90b` (first commit)
- hotcent 2.1.0 (git `685bf7de3297d7b7465e217a6c62a6cec6e02043`), pylibxc 7.1.2, conda-forge
- The electronic part comes from hotcent with PTBP confinement parameters; the repulsive spline comes from physrep (see `tools/physrep.py`)
- Format: simple SKF, `0.02, 719` grid; the polynomial repulsive coefficients are all zero

```
2a417b0efd931de078c26621d8fdda3596a114847a14c148b6976b3bbde027c9  C-C.skf
7239158d51e6ecb3e3014715204f0682b00684e9b1b8e227ec6c0cfc6804ae98  C-Cu.skf
77562d5eecbf1519cd39c1d16bc49aee00637c1839a24b45172f7677fba52545  C-H.skf
fbd7ee768cd013955de76c8163cb616edbd051cbd269483557d22a53132f7deb  C-O.skf
e4c43f175b9346ab6bca7aeb0f44501ed8f1b0bebce914711e8c18bcf0918039  Cu-C.skf
d4520c857b5c217062f0391d3f255c7868c079f667e3ffbcd8c4ad924e9d1e12  Cu-Cu.skf
0744b9537f6c8d0237efbd84ca2449303bea48c7b186e2888be021edcf1a499f  Cu-H.skf
d3bf407f8d3e060be8b7584e585a0e64d4f68b48190c6977b3383108e9e88964  Cu-O.skf
1d20c1d4163ead29934c13abe014578f01827c5d1a819b83f50ccdcf65a650ab  H-C.skf
341ee228ce9eec02b774df8a86474565da6484341938b466757c698ae00fdfbf  H-Cu.skf
bda07f1b30ced641fe44538bcefd52e09ff97f36e737dfb1ca5b3808a7649778  H-H.skf
3e67700111ea0eadba5d75212688de08d63318c27b60a3e8d36af1dcfa955519  H-O.skf
afa99c160898719078718e5fa6371a9e3a40da231441ecf422fa26e5bd6d3e05  O-C.skf
939a8935d2ead1159721ed4e41d29d285a2ef5657f708ac8fa0e43a63150d8cc  O-Cu.skf
c8f50b470190dd83c3ec92116db3a948058952d3aa0b1ec4389e3c3189d854bd  O-H.skf
59e0150a8c586b11cc581e828f83d2ecf9bccc910c6192ad43ab9f001b95aac1  O-O.skf
```

## `orbit/` — H, O, Li

These are copies of `test/orbit/*.skf` from this repository (PTBP-style files with spline repulsives).
There is no Li–O pair, so Li can only be combined with H.

```
3e3fafe6e75523b0d693043b9c72b0d81d9be5231ea76fa3ea822a96627f69c8  H-H.skf
4799b9b4e12662c024c96a457d83fefb38ce97d2743ac40341924d19402b1777  H-Li.skf
7e314b63436f8c046f86cf696eb5644452b1c9bbc6263d886ef76b56efe874bd  H-O.skf
43b7c1f8746044e8b30ede7cf1fa98ccdbf59bd07decda0fd384a0b97c7548b5  Li-H.skf
6e6cbd2eb8e1173f10df93cc1cb24e5b4e927e9d3186ff51a5935af2a8722b86  Li-Li.skf
55253ae92db1a6a3c508071544267592e20be798853cbab59af9b6e7d33d3028  O-H.skf
f2412d7d1a09cc6993f43e2fc88904b3d04816fd0a337071b3a25b27a029226a  O-O.skf
```

## `hotcent_gdo/` — Gd (s, p, d, f basis) and O, for f orbitals (milestone M2g)

Generated with `python -m pydftb.validation.make_f_set` (the script holds every setting):

- hotcent 2.1.0 (git `685bf7de3297d7b7465e217a6c62a6cec6e02043`), pylibxc 7.1.2, conda-forge
- PTBP's recipe: PBE, scalar-relativistic atoms, density superposition, power confinement
  (p = 2), grid 0.4 + 0.02 n Bohr (700 points)
- O: PTBP's published confinement (r_wave 2.011, r_dens 2.664 Bohr) and atomic values; the
  integrals of `O-O.skf` are identical to those of `ptbp_hcocu/O-O.skf`
- Gd: valence 6s 6p 5d 4f, `[Xe] 4f7 5d1 6s2`, r_wave 6.0 and r_dens 9.0 Bohr (chosen for
  testing); on-site energies, Hubbard values and spin constants of the free atom (hotcent's
  analytical U and W)
- Files with Gd are in the extended format (`@`), `O-O.skf` in the simple format
- No repulsive: all polynomial coefficients are zero and there is no spline, so DFTB+ needs
  `PolynomialRepulsive = SetForAll { Yes }`. For testing only, not a physical parametrisation.

```
5be335af1f5564daa60aa799f97859d35febd6b2c6b5bcb463dd64fad5ac33f4  Gd-Gd.skf
a3b7f24f9c20c0e04e5e0214b27f4c28b33ad3218d829134e3ee7410dcd73826  Gd-O.skf
85601a91821131131037583c9bfd05e26842d15e6a7fab55a8d85706848c5a35  O-Gd.skf
d0011939f2ef459da31b00294df7a090bbfb37bd455a17e7e67e3089536936c8  O-O.skf
```
