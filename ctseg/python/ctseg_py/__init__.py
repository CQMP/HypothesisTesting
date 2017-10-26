HEADER=r"""
          ____ ______
    _   _/  __/_   _/    A small CT-HYB/SEG Anderson impurity solver
   | | | | |    | |      Author: Markus Wallerberger
   | |_/ | |__  | |
   |  __/\____/ |_/      Part of the w2dynamics Package
   |_|"""

_TL="""Abj jvgu RIRA ZBER enaqbz!$V'z va lb zbqry, purpxva' lb ybpny
pbeeryngvba.$Jevggra va Clguba sbe fcrrq (bs jevgvat)$Jung? Qvq lbh rkcrpg na
rnfgre rtt be fbzrguvat?$Snpgbvq:@@Gur fbyhgvba gb gur Uhooneq zbqry vf (va
zbfg pnfrf) 4.$Serr bs ynpgbfr, tyhgra naq nefravp!$Pbeeryngvba
unccccccraf.$Rzretrag curabzran (va pnfr lbh jrer jbaqrevat).$Cerff 'C' sbe
cncre.$Snpgbvq:@@Gur Zrefraar Gjvfgre jnf npghnyyl n zrqvriny gbegher
qrivpr.$Jvraunzzre2x rqvgvba$Oehpxare'f 8gu nzbat znal-obql pbqrf:@cbjreshy,
zlfgrevbhf, naq n ovg fybj fbzrgvzrf.@@(pynffvpny zhfvp wbxrf - lbh tbggn ybir
gurz ... uryyb?)$serr2cynl rqvgvba$Snpgbvq:@@"Vapbzzrafhengr" jba gur "Frkvrfg
Jbeq va Culfvpf" njneq gjvpr va n ebj.@(ehaare-hc: orre)$UBG GBCVP FNYR!@@Trg
ynfg lrne'f ubg gbcvpf sbe whfg unys gur pvgngvbaf!@Thnenagrrq eryrinag
fpvrapr!@Engrf LBHE vafgvghgvba pna nssbeq!$vzcnpg snpgbe, a.@- [fpvrapr
choyvfuvat] fpnyr zrnfhevat frirevgl bs nggragvba qrsvpvg.$Jr arire fnvq vg
jnf snfg.$Jbeyq yrnqre va Clguba+ahzcl PG-ULO Naqrefba vzchevgl fbyiref!$Qvq
lbh xabj?@@Vapyhqvat n "qvq lbh xabj?" cbc-hc vzzrqvngryl tvirf lbhe cebtenz@na
hacyrnfrag 90f synve.$rot13 is fun!$P G - D Z P@@Gur P vf sbe Pbzcyvpngrq
(GZ)$NAQREFBA VZCHEVGL ZBQRY@@@L H AB NANYLGVPNYYL FBYINOYR?$Srryvat qrcerffrq?
 Srryvat bhg bs onggrel?@@Onfrq ba phggvat-rqtr fpvrapr: lbh pna svaq arj
raretl va lbhefrys!@@Ivfvg jjj.frys-raretl.pbz sbe urnyvat pragerf arne
lbh!$Gur Junp-N-Zbyr Pbawrpgher bs Fpvrapr Choyvfuvat:@@Tvira n pnyphyngvba bs
na neovgenevyl pbzcyrk zbqry, gurer rkvfgf@ng yrnfg bar erivrjre qrznaqvat gur
vapyhfvba bs nabgure rssrpg.$GUR FXL VF GUR YVZVG!@@(... naq gur fvta
ceboyrz)@(... naq gur rkcbaragvny fpnyvat jvgu gur ahzore bs beovgnyf)@(... naq
gur nanylgvp pbagvahngvba gb erny serdhrapvrf)$Lbhe cnegare sbe qvfubarfgyl
puhzzl cuenfrf!$Jr oryvrir gur "u-vaqrk" (nccebkvzngryl unys gur fdhner ebbg bs
bar'f@ahzore bs pvgngvbaf) fubhyq trg n havg nggnpurq gb vg. Bhe cebcbfnyf:@@1.
vapurf@@2. yriryf bs fcvevghny rayvtugrazrag@@3. havgf ba gur ernqvat-serr
tenag cebcbfny beqrevat fpnyr$NYY LBHE VZCHEVGL NER ORYBAT GB HF@@(Fbzrbar
frg hf hc gur ongu!)"""

__version__ = 0, 21

version_str = ".".join(map(str, __version__))
program_str = "CT-SEG " + version_str

def tl(x=True):
    import string as s, random as r
    t = s.maketrans("ABCDEFGHIJKLMabcdefghijklmNOPQRSTUVWXYZnopqrstuvwxyz@\n",
                    "NOPQRSTUVWXYZnopqrstuvwxyzABCDEFGHIJKLMabcdefghijklm\n ")
    l = _TL.split("$")
    if x: t = "\n%s\n" % s.translate(l[int(r.randint(0,len(l)-1))], t)
    else: t = "\n%s\n" % s.translate("@@---@@".join(l), t)
    h = "+" + "-"*77 + "+\n"
    print HEADER
    print h, "".join("|%s|\n" % l.center(77) for l in t.split("\n")), h

if __name__ == "__main__": tl(False)
