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

## `cu_types/` — two parameter sets of Cu (`Cu`, `Cu2`), for smooth type switching (milestone M5a)

Generated with `python -m pydftb.validation.make_cu_types_set` (the script holds every setting;
a second run gives byte-identical files):

- hotcent 2.1.0 (git `685bf7de3297d7b7465e217a6c62a6cec6e02043`), pylibxc 7.1.2, conda-forge
- PTBP's recipe (`utils/hotpy.py`): PBE, scalar-relativistic atoms, self-consistent power
  confinement (p = 2) of the valence subshells (r0_w) and of the density (r0_d), density
  superposition, grid 0.4 + 0.02 n Bohr (700 points); PTBP's table values for the on-site
  energies, Hubbard values and occupations, so both types have the same on-site data
- `Cu`: r0_w 2.749, r0_d 12.036 Bohr (PTBP's published Cu confinement); `Cu2`: r0_w 3.0,
  r0_d 12.036 Bohr (looser orbitals)
- Cross files `Cu-Cu2.skf` and `Cu2-Cu.skf`: both directions computed with the wave-function
  range of both atoms, heteronuclear format (no on-site line), as hotcent would do for two
  elements (hotcent 2.1 treats two atoms of one symbol as one atom)
- Repulsive: physrep's spline (`tools/physrep.py`, k_xc = 0) with sigma_rep 0.42 (`Cu`) and 0.45
  (`Cu2`), the cross pair with both atom sizes. For testing only, not a physical
  parametrisation.

```
7ed900b3e4f703d541aee10b0d0df5048092204ba9495fb92cb71c1406b3c4af  Cu-Cu.skf
d6fc694312fd27cc6dd4e9cb9983132176e7742bdfe325a0d6ef97e36bc201f0  Cu-Cu2.skf
6e6959c48fe270ec0e792fdebc6c00f3fd47e9f18bb8a636e78ba37ef5ff0198  Cu2-Cu.skf
5e51be114fd91dd2877193f953e57df2a5a6e8dc154c0916048cf88501a168a6  Cu2-Cu2.skf
```

## `ptbp_hcocu_lc/`, `ptbp_hcocu_cam/`, `ptbp_hcocu_global/` — hybrid functionals (milestone M6h-1)

Generated with `python -m pydftb.validation.make_hybrid_sets` (deterministic: gzip without a time
stamp; a second run gives byte-identical files): copies of `ptbp_hcocu/` with a `RangeSep`
section inserted before `<Documentation>` in every file, as in DFTB+'s ob2-1-1 files:

- `ptbp_hcocu_lc`: H, C, O, Cu with `LC 0.3` (LC-DFTB, omega = 0.3 / Bohr)
- `ptbp_hcocu_cam`: H, C, O with `CAM 0.3 0.19 0.46` (the values of DFTB+'s CAM tests)
- `ptbp_hcocu_global`: H, C, O with `CAM 0.0 0.2 0.0` (a global hybrid: beta = 0)

The integrals, on-site data and repulsives are those of the semi-local `ptbp_hcocu` files:
the long-range exchange is added to a parametrisation that already holds semi-local exchange.
For testing pydftb against DFTB+ only, not a physical parametrisation. The SHA-256 of the
uncompressed files:

`ptbp_hcocu_lc/`:

```
466b1da0599aadd1622c3d16588ac09f568b6c9ab4e538dcbfef2538a098605b  H-H.skf
c24c84231cb80ed922e22d5dd4571640aa7ae20b9294b2c05019baf122f1c946  H-C.skf
f68b93fa05bd427329ae0126d76f5973b732f17a7ead72ca7fbee84ed8addafe  H-O.skf
8d1659e7154f1380edf2442e1b846f9530c01b668de9032d48d9f877eab7be4f  H-Cu.skf
ff1f6fe641768bb259adc508f04e5f60f698a9876b4fec90152db24f551ad24e  C-H.skf
bdce9b8f8484237a77dfec9739f12e4230a2621e240781eb3fc4caa11518ba62  C-C.skf
7fc441d083ae808321384eee47c410fe779b6c4042bdfa040f1aa6c97a349261  C-O.skf
ca2097e8c281e2c911474e9109727d19a33ec72a23f267374f44aa0eb56adc51  C-Cu.skf
cbfe7271cc0e99b89a03eaf24c6ad9f4b8b70e2eb653cd31fd334739cc1cfda2  O-H.skf
96402572482cce60c17209013feae10100829fb3a34b42efdec1ac83173ebd37  O-C.skf
6e1e7d22e464f863a6497c24b5d4407a56f192ca21d4cba389d4f4123711e3d0  O-O.skf
2748c5f9e1617292165b584656cad4a6a63d564472b0e9d1819fdba6f9c51d19  O-Cu.skf
6170214013fd44e795cf7e941530be17846d4997af7dea94c523a81af83d42db  Cu-H.skf
51816a49b9e8094200ebfad0bbc3017c3eff28210042951022b8e1e7ae63f3f8  Cu-C.skf
6509210c8cbcf631bf33df49f347c2fbdde3a85ac69829efddb82581052e8493  Cu-O.skf
42062ec155a31d2c10e948be0624890f8c34c3fb51291b5556a82e60a5a00de7  Cu-Cu.skf
```

`ptbp_hcocu_cam/`:

```
fa67e54fba47f60bb6720980ddb64574982ba9cb2f423cfab31c582d1f15622b  H-H.skf
6f8166cf12670c76c1f383f5cf2aa8929dc51648381860c95e31efaaf90d4de0  H-C.skf
5aa3c3de3c55cded2ed0280ba8d042147b06b6df4fe206409ae76d8f3f7f5f6d  H-O.skf
444a1a8d140e3fbadad1de919624cf700393282f5c26f613d947b1951332eaa5  C-H.skf
d5ceacd37666374025514755ecd5851b6bdc6221dcc1500b201411e210feadf9  C-C.skf
de35b88b0e458ba86cd4367e142330ae4db87cb5a47069370c3bacd258d46550  C-O.skf
09344d5e088ef0c02a547140fe8be5951eaa9670d1dee24d24590c13bc5c7e1e  O-H.skf
ab3a20e49fdc4c627b4e4af6c14d078edd3e22b4a1489e593b9f820e4af30351  O-C.skf
2718f1c6a0c729326bff3bbaf65a9039a0381d15a09d84b20637f399b5a69486  O-O.skf
```

`ptbp_hcocu_global/`:

```
ce15ed15eaf96bb35bfbab79aafc529fd7c3c4c225d875b9dc1fe8edb652533e  H-H.skf
a8019bb44a182a96069f0ac29bc889442eebe6c428a42a813b93b51ab78a4216  H-C.skf
ddf2eb4148d4e0f4e90e8239cb102b4b9062247345e38df62f5cb47105d43c95  H-O.skf
c223a8f6b76d07159692d850723fd5f2f42b8f0b8520b757f39ab68c94f3fe81  C-H.skf
d2d63c33ef40af8eb0e8bc43e16c1c2bf34af8396e4a7657d0403bd7b3cf0858  C-C.skf
b5c96c29ec54d6b2864bc3b72622c42a1a68548126678d142dcc3ecbe139000c  C-O.skf
78353e6846e44bdaf4fe00f9b66c0c6221c09b83e4218aecc66c450ffe0710c4  O-H.skf
9a30d0a3ac51ee07824e824b65947a3440024dc9dfebf627cd1b679c55f9ba23  O-C.skf
a841a144f55cdf1d86d9f9fb8aff124a5f54b17bc96f6fb215be11d1e8cab1d6  O-O.skf
```
