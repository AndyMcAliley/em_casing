'''
A module for modeling electromagnetic fields in a halfspace with a vertical casing

functions:

hankel_J1_140
hankel_J1_201
fii
fij
Zii
Zij
form_gamma_casing
form_A
HED_Ez
HEB_Ez
form_b_analytic
form_b

BEWARE: This module uses an e^(-iwt) convention for A, but an e^(iwt) convention for b.
TODO: Use a consistent sign convention
'''

import numpy as np

mu0 = 4e-7*np.pi
W140 = [-6.76671159511e-14,
        3.39808396836e-13,
        -7.43411889153e-13,
        8.93613024469e-13,
        -5.47341591896e-13,
        -5.84920181906e-14,
        5.20780672883e-13,
        -6.92656254606e-13,
        6.88908045074e-13,
        -6.39910528298e-13,
        5.82098912530e-13,
        -4.84912700478e-13,
        3.54684337858e-13,
        -2.10855291368e-13,
        1.00452749275e-13,
        5.58449957721e-15,
        -5.67206735175e-14,
        1.09107856853e-13,
        -6.04067500756e-14,
        8.84512134731e-14,
        2.22321981827e-14,
        8.38072239207e-14,
        1.23647835900e-13,
        1.44351787234e-13,
        2.94276480713e-13,
        3.39965995918e-13,
        6.17024672340e-13,
        8.25310217692e-13,
        1.32560792613e-12,
        1.90949961267e-12,
        2.93458179767e-12,
        4.33454210095e-12,
        6.55863288798e-12,
        9.78324910827e-12,
        1.47126365223e-11,
        2.20240108708e-11,
        3.30577485691e-11,
        4.95377381480e-11,
        7.43047574433e-11,
        1.11400535181e-10,
        1.67052734516e-10,
        2.50470107577e-10,
        3.75597211630e-10,
        5.63165204681e-10,
        8.44458166896e-10,
        1.26621795331e-09,
        1.89866561359e-09,
        2.84693620927e-09,
        4.26886170263e-09,
        6.40104325574e-09,
        9.59798498616e-09,
        1.43918931885e-08,
        2.15798696769e-08,
        3.23584600810e-08,
        4.85195105813e-08,
        7.27538583183e-08,
        1.09090191748e-07,
        1.63577866557e-07,
        2.45275193920e-07,
        3.67784458730e-07,
        5.51470341585e-07,
        8.26916206192e-07,
        1.23991037294e-06,
        1.85921554669e-06,
        2.78777669034e-06,
        4.18019870272e-06,
        6.26794044911e-06,
        9.39858833064e-06,
        1.40925408889e-05,
        2.11312291505e-05,
        3.16846342900e-05,
        4.75093313246e-05,
        7.12354794719e-05,
        1.06810848460e-04,
        1.60146590551e-04,
        2.40110903628e-04,
        3.59981158972e-04,
        5.39658308918e-04,
        8.08925141201e-04,
        1.21234066243e-03,
        1.81650387595e-03,
        2.72068483151e-03,
        4.07274689463e-03,
        6.09135552241e-03,
        9.09940027636e-03,
        1.35660714813e-02,
        2.01692550906e-02,
        2.98534800308e-02,
        4.39060697220e-02,
        6.39211368217e-02,
        9.16763946228e-02,
        1.28368795114e-01,
        1.73241920046e-01,
        2.19830379079e-01,
        2.51193131178e-01,
        2.32380049895e-01,
        1.17121080205e-01,
        -1.17252913088e-01,
        -3.52148528535e-01,
        -2.71162871370e-01,
        2.91134747110e-01,
        3.17192840623e-01,
        -4.93075681595e-01,
        3.11223091821e-01,
        -1.36044122543e-01,
        5.12141261934e-02,
        -1.90806300761e-02,
        7.57044398633e-03,
        -3.25432753751e-03,
        1.49774676371e-03,
        -7.24569558272e-04,
        3.62792644965e-04,
        -1.85907973641e-04,
        9.67201396593e-05,
        -5.07744171678e-05,
        2.67510121456e-05,
        -1.40667136728e-05,
        7.33363699547e-06,
        -3.75638767050e-06,
        1.86344211280e-06,
        -8.71623576811e-07,
        3.61028200288e-07,
        -1.05847108097e-07,
        -1.51569361490e-08,
        6.67633241420e-08,
        -8.33741579804e-08,
        8.31065906136e-08,
        -7.53457009758e-08,
        6.48057680299e-08,
        -5.37558016587e-08,
        4.32436265303e-08,
        -3.37262648712e-08,
        2.53558687098e-08,
        -1.81287021528e-08,
        1.20228328586e-08,
        -7.10898040664e-09,
        3.53667004588e-09,
        -1.36030600198e-09,
        3.52544249042e-10,
        -4.53719284366e-11]
ab140 = 10**(-7.91001919000+np.arange(140)*0.0879671439570)

Wab201 = np.array([[4.1185887075357082e-06,  1.5020099209519960e-03,  4.7827871332506182e-10],
    [4.6623077830484039e-06, -1.0381698214761684e-02, -2.9784175503440788e-09],
    [5.2778064058937756e-06,  3.6840860097595164e-02,  9.7723832770897220e-09],
    [5.9745606155328603e-06, -8.9903380392274704e-02, -2.2382340996085809e-08],
    [6.7632974390298035e-06,  1.7082286536833860e-01,  4.0446774329470848e-08],
    [7.6561600412698171e-06, -2.7115749656836280e-01, -6.1734815854553919e-08],
    [8.6668946776271221e-06,  3.7649328091859574e-01,  8.3293912185608189e-08],
    [9.8110623273521872e-06, -4.7220778569122657e-01, -1.0249453502284080e-07],
    [1.1106278265924788e-05,  5.4778211089647089e-01,  1.1780779749909977e-07],
    [1.2572483264760042e-05, -5.9823516853035097e-01, -1.2870061460834850e-07],
    [1.4232250593579901e-05,  6.2345791612185331e-01,  1.3559243438349901e-07],
    [1.6111133551969887e-05, -6.2650436648257724e-01, -1.3921010821521872e-07],
    [1.8238058880617237e-05,  6.1197225351173345e-01,  1.4065745722769670e-07],
    [2.0645772109076137e-05, -5.8470173866140451e-01, -1.4074881908375281e-07],
    [2.3371341696506312e-05,  5.4911055789616858e-01,  1.4051720878600928e-07],
    [2.6456729726989950e-05, -5.0878861684860843e-01, -1.4040746687777830e-07],
    [2.9949437945688260e-05,  4.6652106345430200e-01,  1.4127886061686993e-07],
    [3.3903239082024059e-05, -4.2426478870289325e-01, -1.4315595655055356e-07],
    [3.8379004719130766e-05,  3.8339538911933357e-01,  1.4689283208027915e-07],
    [4.3445642455208216e-05, -3.4472344791723936e-01, -1.5210916706348747e-07],
    [4.9181156785051293e-05,  3.0876891466890510e-01,  1.5989801550138741e-07],
    [5.5673850034776868e-05, -2.7570439368364807e-01, -1.6940918407911260e-07],
    [6.3023681838995322e-05,  2.4562331000974616e-01,  1.8227089415749844e-07],
    [7.1343808090545215e-05, -2.1839207265400126e-01, -1.9695856878603369e-07],
    [8.0762323056016586e-05,  1.9393194321380827e-01,  2.1603952427106760e-07],
    [9.1424231478173270e-05, -1.7198162962644575e-01, -2.3691320619292838e-07],
    [1.0349368102719458e-04,  1.5242270280410272e-01,  2.6369843208466607e-07],
    [1.1715648947091054e-04, -1.3494945901825492e-01, -2.9202021404039016e-07],
    [1.3262300547161834e-04,  1.1946763189654602e-01,  3.2852445086324662e-07],
    [1.5013134705348249e-04, -1.0565870880444576e-01, -3.6589094553627693e-07],
    [1.6995106759902750e-04,  9.3482355548033913e-02,  4.1487501036863030e-07],
    [1.9238730581535294e-04, -8.2612525279836244e-02, -4.6327136995173986e-07],
    [2.1778548356175115e-04,  7.3077763526174969e-02,  5.2852697369750518e-07],
    [2.4653662386513240e-04, -6.4536481281701225e-02, -5.9034983719954077e-07],
    [2.7908337099788406e-04,  5.7096310621587334e-02,  6.7710211611560898e-07],
    [3.1592680530155527e-04, -5.0384859605213800e-02, -7.5512942807901028e-07],
    [3.5763415767542709e-04,  4.4599557396733047e-02,  8.7062473466409799e-07],
    [4.0484754250000483e-04, -3.9316995063444334e-02, -9.6788530918130449e-07],
    [4.5829384344501882e-04,  3.4838544754296139e-02,  1.1222658353725904e-06],
    [5.1879590436097246e-04, -3.0664946477420088e-02, -1.2417228058919743e-06],
    [5.8728519754599128e-04,  2.7221072240291456e-02,  1.4493467036899140e-06],
    [6.6481616442494776e-04, -2.3901586108434601e-02, -1.5932456559208080e-06],
    [7.5258244942583781e-04,  2.1281566646364738e-02,  1.8747045814274419e-06],
    [8.5193527698548301e-04, -1.8612484595521447e-02, -2.0433320340041385e-06],
    [9.6440425461164468e-04,  1.6655386303461400e-02,  2.4285695351967146e-06],
    [1.0917209222795108e-03, -1.4472420076504743e-02, -2.6179926456520528e-06],
    [1.2358454107222641e-03,  1.3057747649606617e-02,  3.1511729661391781e-06],
    [1.3989966190390965e-03, -1.1226286001060300e-02, -3.3492412032881268e-06],
    [1.5836863762264436e-03,  1.0266818557930830e-02,  4.0964186613549959e-06],
    [1.7927581125735061e-03, -8.6738022421653585e-03, -4.2758275751239154e-06],
    [2.0294306362957340e-03,  8.1103368670848164e-03,  5.3371197552207799e-06],
    [2.2973476893787268e-03, -6.6573628142085938e-03, -5.4435423773626499e-06],
    [2.6006340455800008e-03,  6.4551189810889585e-03,  6.9725766921672099e-06],
    [2.9439590142573526e-03, -5.0524015117839084e-03, -6.9045562968161745e-06],
    [3.3326083277104117e-03,  5.1989014270506263e-03,  9.1397025436977438e-06],
    [3.7725655187922052e-03, -3.7597430078747883e-03, -8.7148373033635960e-06],
    [4.2706040416570154e-03,  4.2640545165292636e-03,  1.2029590806160379e-05],
    [4.8343915539089625e-03, -2.6994965688974474e-03, -1.0927976968519436e-05],
    [5.4726079656493079e-03,  3.5928000560587229e-03,  1.5912526719455299e-05],
    [6.1950790728715292e-03, -1.8061291289989019e-03, -1.3582559661331659e-05],
    [7.0129278325854246e-03,  3.1436470898586703e-03,  2.1176226828087565e-05],
    [7.9387456086585442e-03, -1.0244200001620755e-03, -1.6678205993448338e-05],
    [8.9867860248264830e-03,  2.8888297674091743e-03,  2.8384979250408712e-05],
    [1.0173184409377162e-02, -3.0605070741955765e-04, -2.0132088397797457e-05],
    [1.1516206210016314e-02,  2.8125907445050334e-03,  3.8372045311188108e-05],
    [1.3036528203437736e-02,  3.9337862066047565e-04, -2.3702184335455945e-05],
    [1.4757556829019875e-02,  2.9102041487017727e-03,  5.2385308850007940e-05],
    [1.6705788547622769e-02,  1.1170884769911178e-03, -2.6854373943701261e-05],
    [1.8911217773465227e-02,  3.1876763415953362e-03,  7.2318581557902158e-05],
    [2.1407798659484324e-02,  1.9097806429850762e-03, -2.8535361884516687e-05],
    [2.4233967845691123e-02,  3.6621020321465411e-03,  1.0108123118106823e-04],
    [2.7433236218606032e-02,  2.8203783587849949e-03, -2.6788477540644352e-05],
    [3.1054858792331431e-02,  4.3626885599390430e-03,  1.4319185407094621e-04],
    [3.5154593024557458e-02,  3.9050024915686802e-03, -1.8108424211338017e-05],
    [3.9795557242315927e-02,  5.3324923486866077e-03,  2.0573561552327273e-04],
    [4.5049202393557801e-02,  5.2303399394270488e-03,  3.6361648565843316e-06],
    [5.0996412085361466e-02,  6.6309348458396984e-03,  2.9991264692859609e-04],
    [5.7728747844643331e-02,  6.8775540687037451e-03,  4.8993332079278846e-05],
    [6.5349858773045680e-02,  8.3371689988226087e-03,  4.4354733854670903e-04],
    [7.3977077298642363e-02,  8.9468564745410102e-03,  1.3589101811995494e-04],
    [8.3743225592195963e-02,  1.0554326601376745e-02,  6.6515823521273764e-04],
    [9.4798660459030126e-02,  1.1562765604509096e-02,  2.9451991608624389e-04],
    [1.0731358818908403e-01,  1.3414536265660639e-02,  1.0105553806271136e-03],
    [1.2148068500391276e-01,  1.4879837399735491e-02,  5.7533964792254050e-04],
    [1.3751806344428075e-01,  1.7084241336519569e-02,  1.5535077418254303e-03],
    [1.5567263036799733e-01,  1.9088092654441269e-02,  1.0621193133794828e-03],
    [1.7622388825676111e-01,  2.1768513715084332e-02,  2.4128970258747457e-03],
    [1.9948823835583873e-01,  2.4416120601223806e-02,  1.8929698186109245e-03],
    [2.2582385189647586e-01,  2.7711234465350204e-02,  3.7800177772191607e-03],
    [2.5563618439697788e-01,  3.1127164234404547e-02,  3.2937343278959356e-03],
    [2.8938421793905067e-01,  3.5184140220277452e-02,  5.9612179800391544e-03],
    [3.2758752752368947e-01,  3.9497913113638941e-02,  5.6295935320552241e-03],
    [3.7083428029819565e-01,  4.4449762157204989e-02,  9.4422526803168080e-03],
    [4.1979029080811425e-01,  4.9758433269983887e-02,  9.4810228247137925e-03],
    [4.7520927168614446e-01,  5.5667457114992429e-02,  1.4979159139973408e-02],
    [5.3794443759467447e-01,  6.1949790434151823e-02,  1.5745093424331037e-02],
    [6.0896164107289685e-01,  6.8682064905571841e-02,  2.3708966370000140e-02],
    [6.8935424252422239e-01,  7.5616743950118387e-02,  2.5740590762136872e-02],
    [7.8035994327803426e-01,  8.2584983563572259e-02,  3.7232782843117498e-02],
    [8.8337984088275090e-01,  8.9193635262987042e-02,  4.1225008900614292e-02],
    [1.0000000000000000e+00,  9.4874686942005279e-02,  5.7507103212277359e-02],
    [1.1320158709991752e+00,  9.8891681869094042e-02,  6.4044642846235691e-02],
    [1.2814599321940212e+00,  1.0004294654495730e-01,  8.6091796551857253e-02],
    [1.4506329812931589e+00,  9.7016844329802857e-02,  9.4717139804457451e-02],
    [1.6421395578187052e+00,  8.7789596149914384e-02,  1.2172497389177185e-01],
    [1.8589280418463421e+00,  7.0509767592855419e-02,  1.2853597000398900e-01],
    [2.1043360464154781e+00,  4.2778853484849347e-02,  1.5450777327408322e-01],
    [2.3821418024579781e+00,  3.5584532926218175e-03,  1.4755964090969320e-01],
    [2.6966223273530128e+00, -4.7210453264879347e-02,  1.5621399202016978e-01],
    [3.0526192726543444e+00, -1.0489787743225988e-01,  1.1147620703185755e-01],
    [3.4556134647626755e+00, -1.6020950407348281e-01,  7.7489831356083380e-02],
    [3.9118092861497971e+00, -1.9459781573132096e-01, -2.7628266850147711e-02],
    [4.4282301962435247e+00, -1.8490774599542381e-01, -1.0198730178317840e-01],
    [5.0128268625854631e+00, -1.0754165020025190e-01, -2.2039889971111640e-01],
    [5.6745995670177445e+00,  3.6037727487476613e-02, -2.1185762869925318e-01],
    [6.4237367714291338e+00,  1.9759013047489976e-01, -1.6052415083152241e-01],
    [7.2717719763787807e+00,  2.6132313321851336e-01,  9.1649025798681089e-02],
    [8.2317612875478190e+00,  1.1713996822458939e-01,  2.3792823877700942e-01],
    [9.3184844237807365e+00, -1.8758779281301441e-01,  2.6075777853738125e-01],
    [1.0548672261378394e+01, -3.0238114997462151e-01, -1.5662188259001042e-01],
    [1.1941264417849103e+01,  4.8163135684567732e-02, -2.8932081756330175e-01],
    [1.3517700840802913e+01,  3.6399529664885466e-01,  1.3148519116247689e-02],
    [1.5302251891207787e+01, -1.4910233461562913e-01,  4.2691302759079564e-01],
    [1.7322392002874359e+01, -2.6373490348543854e-01, -4.0005050006489040e-01],
    [1.9609222670922968e+01,  4.0362661807718703e-01,  1.1513789407450359e-01],
    [2.2197951281441636e+01, -3.1409794650104578e-01,  9.3748244358717620e-02],
    [2.5128433154258410e+01,  1.8179369405131079e-01, -1.6037231301955096e-01],
    [2.8445785143962375e+01, -9.0738718042631769e-02,  1.5071857939129532e-01],
    [3.2201080245997971e+01,  4.2946487545160242e-02, -1.2120369075996129e-01],
    [3.6452133901787732e+01, -2.0586135807067835e-02,  9.4110656079982341e-02],
    [4.1264394108610787e+01,  1.0392667161913182e-02, -7.3742238434584328e-02],
    [4.6711949038112273e+01, -5.6117848068723023e-03,  5.9038567576124905e-02],
    [5.2878667676447755e+01,  3.2402025511569896e-03, -4.8288117528475852e-02],
    [5.9859491047029913e+01, -1.9858724388273777e-03,  4.0197054299576880e-02],
    [6.7761893895170928e+01,  1.2807317326135252e-03, -3.3919787720641081e-02],
    [7.6707539338295589e+01, -8.6253791756068252e-04,  2.8918247156763971e-02],
    [8.6834151956244213e+01,  6.0296590782143555e-04, -2.4845271759013743e-02],
    [9.8297638159222487e+01, -4.3548936996943465e-04,  2.1470449751150148e-02],
    [1.1127448647797397e+02,  3.2375891570874245e-04, -1.8635828020057092e-02],
    [1.2596448473034971e+02, -2.4698212240059978e-04,  1.6229579362363859e-02],
    [1.4259379589698909e+02,  1.9279062274925971e-04, -1.4170085406786529e-02],
    [1.6141844006140866e+02, -1.5357911509105972e-04,  1.2396084121011890e-02],
    [1.8272823602144376e+02,  1.2453787849367440e-04, -1.0860414401084047e-02],
    [2.0685126325595743e+02, -1.0255126402954421e-04,  9.5259444245356633e-03],
    [2.3415891294197226e+02,  8.5558482209476271e-05, -8.3628577447233381e-03],
    [2.6507160578622688e+02, -7.2170928476334345e-05,  7.3468029551253195e-03],
    [3.0006526470124555e+02,  6.1436863283080017e-05, -6.4576043210966359e-03],
    [3.3967864197737873e+02, -5.2693349406473615e-05,  5.6783439955994880e-03],
    [3.8452161375783919e+02,  4.5471255142623729e-05, -4.9946949167265437e-03],
    [4.3528456951608860e+02, -3.9433284158653591e-05,  4.3944258108608806e-03],
    [4.9274904109325632e+02,  3.4332971164795619e-05, -3.8670264019660858e-03],
    [5.5779973493719069e+02, -2.9987212220165472e-05,  3.4034180355556670e-03],
    [6.3143815278803334e+02,  2.6257657435614391e-05, -2.9957260668529964e-03],
    [7.1479801051045558e+02, -2.3037978256349448e-05,  2.6370977166248776e-03],
    [8.0916269245647072e+02,  2.0245071331016652e-05, -2.3215540117372982e-03],
    [9.1598501008114988e+02, -1.7812925644382519e-05,  2.0438677474690805e-03],
    [1.0369095690092008e+03,  1.5688305607849465e-05, -1.7994616759226389e-03],
    [1.1737980889093292e+03, -1.3827679492110470e-05,  1.5843226896713463e-03],
    [1.3287580659938624e+03,  1.2195005442258958e-05, -1.3949288614414647e-03],
    [1.5041752194232211e+03, -1.0760110818279243e-05,  1.2281869708886250e-03],
    [1.7027502211507524e+03,  9.4974857670959145e-06, -1.0813786997088304e-03],
    [1.9275402746900081e+03, -8.3853711343938338e-06,  9.5211407460757294e-04],
    [2.1820061829391980e+03,  7.4050612540713192e-06, -8.3829103020448140e-04],
    [2.4700656297055034e+03, -6.5403682860228500e-06,  7.3806018220987763e-04],
    [2.7961534952362003e+03,  5.7772102413267325e-06, -6.4979406671247244e-04],
    [3.1652901343571971e+03, -5.1032933218796731e-06,  5.7206022901230412e-04],
    [3.5831586684094545e+03,  4.5078641320869195e-06, -5.0359764543483573e-04],
    [4.0561924809477755e+03, -3.9815111951817114e-06,  4.4329604122300101e-04],
    [4.5916742642604040e+03,  3.5159993210292339e-06, -3.9017773206623073e-04],
    [5.1978481416012310e+03, -3.1041249680128287e-06,  3.4338166974098888e-04],
    [5.8840465913361650e+03,  2.7395854235553632e-06, -3.0214941633163506e-04],
    [6.6608341270911405e+03, -2.4168587823583312e-06,  2.6581280850704716e-04],
    [7.5401699459601114e+03,  2.1310948134441218e-06, -2.3378310479588391e-04],
    [8.5355920488578286e+03, -1.8780185742021288e-06,  2.0554143583887405e-04],
    [9.6624256676814348e+03,  1.6538488252707735e-06, -1.8063040107732216e-04],
    [1.0938019208165191e+04, -1.4552318768651144e-06,  1.5864667598176302e-04],
    [1.2382011340936813e+04,  1.2791890865306748e-06, -1.3923451235168759e-04],
    [1.4016633352832258e+04, -1.1230741959148155e-06,  1.2208003098625932e-04],
    [1.5867051413382505e+04,  9.8453630538596172e-07, -1.0690622166177854e-04],
    [1.7961754025908867e+04, -8.6148574493190796e-07,  9.3468580362568137e-05],
    [2.0332990628312182e+04,  7.5206242699701732e-07, -8.1551328581234316e-05],
    [2.3017268096126892e+04, -6.5460810334682633e-07,  7.0964174631531072e-05],
    [2.6055912791858576e+04,  5.6764451596029765e-07, -6.1539592468666665e-05],
    [2.9495706813754354e+04, -4.8985884555184997e-07,  5.3130609116145443e-05],
    [3.3389608239508460e+04,  4.2009674106879252e-07, -4.5609105983126460e-05],
    [3.7797566453568354e+04, -3.5736212641225517e-07,  3.8864648584181207e-05],
    [4.2787445110585410e+04,  3.0082215969939671e-07, -3.2803856352344075e-05],
    [4.8436066944688770e+04, -2.4981510481639669e-07,  2.7350296775876261e-05],
    [5.4830396510166145e+04,  2.0385823466866512e-07, -2.2444816150805301e-05],
    [6.2068879062685897e+04, -1.6265189071584773e-07,  1.8046076281584239e-05],
    [7.0262956194088882e+04,  1.2607416700611311e-07, -1.4130826937491561e-05],
    [7.9538781555028458e+04, -9.4158417913450858e-08,  1.0693106849359383e-05],
    [9.0039163080228551e+04,  6.7043911217063425e-08, -7.7412053314530284e-06],
    [1.0192576161830177e+05, -4.4891090827293947e-08,  5.2910576443698300e-06],
    [1.1538157981559623e+05,  2.7761325666544702e-08, -3.3552268362550323e-06],
    [1.3061377957221285e+05, -1.5480404355710375e-08,  1.9282956206367452e-06],
    [1.4785687144693290e+05,  7.5327300141098751e-09, -9.7253712572058755e-07],
    [1.6737632511421290e+05, -3.0524770418657847e-09,  4.1100807632959352e-07],
    [1.8947265645880660e+05,  9.5877856096830783e-10, -1.3553176263207053e-07],
    [2.1448605423174356e+05, -2.0575286298055636e-10,  3.0748587523233524e-08],
    [2.4280161749832361e+05,  2.2414416956474645e-11, -3.5668195345476294e-09]])


def hankel_J1_140(function, r):
    '''
    Compute Hankel transform of order 1:
    f(r) = integral 0 -> inf (function(lamda) * J1(r*lamda) * d lamda)
    Follows Guptasarma and Singh (1997)
    '''
    # a = -7.91001919000
    # s = 0.0879671439570
    # lamda = 10**(a+np.arange(140)*s)/r
    lamda = ab140/r
    K = function(lamda)
    f = np.dot(K,W140)
    return f/r

def hankel_J1_201(function, r):
    '''
    Compute Hankel transform of order 1:
    f(r) = integral 0 -> inf (function(lamda) * J1(r*lamda) * d lamda)
    Follows Key 2012
    '''
    lamda = Wab201[:,0]/r
    K = function(lamda)
    f = np.dot(K,Wab201[:,2])
    return f/r

def fii(lamda,z,dz,frequency,conductivity):
    '''
    function in Hankel transform for gamma_ii elements
    lamda is an array of lamda values
    z is vertical coord of center of ith segment
    dz is length of ith segment
    '''
    # mu0 = 4*np.pi*1e-7
    s = np.sqrt(lamda**2 - 1j*2*np.pi*frequency*mu0*conductivity)
    result = 2*np.exp(-s*dz/2)
    result -= 2
    result -= np.exp(-s*(2*z+dz/2))
    result += np.exp(-s*(2*z-dz/2))
    result *= lamda**2/s**2
    return result

def fij(lamda,zi,zj,dz,frequency,conductivity):
    '''
    function in Hankel transform for gamma_ij elements
    lamda is an array of lamda values
    zi is vertical coord of center of ith segment
    zj is vertical coord of center of jth segment
    dz is length of segments
    '''
    # mu0 = 4*np.pi*1e-7
    s = np.sqrt(lamda**2 - 1j*2*np.pi*frequency*mu0*conductivity)
    result = np.exp(-s*(abs(zi-zj)+dz/2))
    result -= np.exp(-s*(abs(zi-zj)-dz/2))
    result -= np.exp(-s*(zi+zj+dz/2))
    result += np.exp(-s*(zi+zj-dz/2))
    result *= lamda**2/s**2
    return result

def Gii(zi,
        dz,
        frequency=0.125,
        background_conductivity=0.18,
        outer_radius=0.1095,
        inner_radius=0.1095-0.0134):
    '''
    Form ith diagonal of integrated Green's matrix to solve for casing current densities
    Follows Tang et al., 2015
    Uses e^(-iwt) convention
    '''
    funii = lambda x:fii(x,zi,dz,frequency,background_conductivity)
    result = outer_radius*hankel_J1_201(funii,outer_radius)
    result -= inner_radius*hankel_J1_201(funii,inner_radius)
    result = -result/2/background_conductivity
    return result

def Zii(zi,
        dz,
        frequency=0.125,
        background_conductivity=0.18,
        casing_conductivity=1.0e7,
        outer_radius=0.1095,
        inner_radius=0.1095-0.0134):
    '''
    Form ith diagonal of coefficient matrix to solve for casing current densities
    Follows Tang et al., 2015
    Uses e^(-iwt) convention
    '''
    gamma_ii = Gii(zi,
                   dz,
                   frequency=frequency,
                   background_conductivity=background_conductivity,
                   outer_radius=outer_radius,
                   inner_radius=inner_radius)
    z_ii = 1/casing_conductivity - gamma_ii
    return z_ii

def Aii_old(zi,
            dz,
            frequency=0.125,
            background_conductivity=0.18,
            casing_conductivity=1.0e7,
            outer_radius=0.1095,
            inner_radius=0.1095-0.0134):
    '''
    Form ith diagonal of coefficient matrix to solve for casing current densities
    Follows Tang et al., 2015
    Uses e^(-iwt) convention
    '''
    funii = lambda x:fii(x,zi,dz,frequency,background_conductivity)
    result = outer_radius*hankel_J1_140(funii,outer_radius)
    result -= inner_radius*hankel_J1_140(funii,inner_radius)
    result = 1/casing_conductivity + result/2/background_conductivity
    return result

def Gij(zi,
        zj,
        dz,
        frequency=0.125,
        background_conductivity=0.18,
        outer_radius=0.1095,
        inner_radius=0.1095-0.0134):
    '''
    Form i,jth element of coefficient matrix
    Follows Tang et al., 2015
    Uses e^(-iwt) convention
    '''
    funij = lambda x:fij(x,zi,zj,dz,frequency,background_conductivity)
    result = outer_radius*hankel_J1_201(funij,outer_radius)
    result -= inner_radius*hankel_J1_201(funij,inner_radius)
    result = -result/2/background_conductivity
    return result

def Zij(zi,
        zj,
        dz,
        frequency=0.125,
        background_conductivity=0.18,
        casing_conductivity=1.0e7,
        outer_radius=0.1095,
        inner_radius=0.1095-0.0134):
    '''
    Form i,jth element of coefficient matrix
    Follows Tang et al., 2015
    Uses e^(-iwt) convention
    '''
    gamma_ij = Gij(zi,
                   zj,
                   dz,
                   frequency=0.125,
                   background_conductivity=0.18,
                   outer_radius=0.1095,
                   inner_radius=0.1095-0.0134)
    z_ij = -gamma_ij
    return z_ij

def Aij_old(zi,
            zj,
            dz,
            frequency=0.125,
            background_conductivity=0.18,
            casing_conductivity=1.0e7,
            outer_radius=0.1095,
            inner_radius=0.1095-0.0134):
    '''
    Form i,jth element of coefficient matrix
    Follows Tang et al., 2015
    Uses e^(-iwt) convention
    '''
    funij = lambda x:fij(x,zi,zj,dz,frequency,background_conductivity)
    result = outer_radius*hankel_J1_140(funij,outer_radius)
    result -= inner_radius*hankel_J1_140(funij,inner_radius)
    result /= 2*background_conductivity
    return result

def form_gamma_casing(frequency=0.125,
                      background_conductivity=0.18,
                      outer_radius=0.1095,
                      inner_radius=0.1095-0.0134,
                      casing_length=1365,
                      num_segments=280,
                      **kwargs):
    '''
    Form integrated Green's tensor matrix to solve for casing current densities
    From Tang et al, 2015
    Return conjugate to convert to e^(iwt) convention

    kwargs are unused
    '''
    dz = casing_length/num_segments
    zs = dz*(np.arange(num_segments)+0.5)
    G = np.ones((num_segments,num_segments))*1j
    #TODO: vectorize or parallelize
    #TODO: skip half of Gij computations (symmetry)
    for ii in np.arange(num_segments):
        zi = zs[ii]
        for jj in np.arange(num_segments):
            zj = zs[jj]
            if ii==jj:
                G[ii,jj] = Gii(zi,
                               dz,
                               frequency=frequency,
                               background_conductivity=background_conductivity,
                               outer_radius=outer_radius,
                               inner_radius=inner_radius
                              )
            else:
                G[ii,jj] = Gij(zi,
                               zj,
                               dz,
                               frequency=frequency,
                               background_conductivity=background_conductivity,
                               outer_radius=outer_radius,
                               inner_radius=inner_radius
                              )
    return np.conj(G)

def form_A(frequency=0.125,
           background_conductivity=0.18,
           casing_conductivity=1.0e7,
           outer_radius=0.1095,
           inner_radius=0.1095-0.0134,
           casing_length=1365,
           num_segments=280,
           **kwargs):
    '''
    Form coefficient matrix to solve for casing current densities
    for a single casing
    Follows Tang et al., 2015
    Uses e^(iwt) time dependence

    kwargs are unused
    '''
    # dz = casing_length/num_segments
    # zs = dz*(np.arange(num_segments)+0.5)
    # Z = np.ones((num_segments,num_segments))*1j
    #TODO: vectorize or parallelize
    #TODO: skip half of Zij computations (symmetry)

    # for ii in np.arange(num_segments):
    #     zi = zs[ii]
    #     for jj in np.arange(num_segments):
    #         zj = zs[jj]
    #         if ii==jj:
    #             Z[ii,jj] = Zii(zi,
    #                            dz,
    #                            frequency=frequency,
    #                            background_conductivity=background_conductivity,
    #                            casing_conductivity=casing_conductivity,
    #                            outer_radius=outer_radius,
    #                            inner_radius=inner_radius
    #                           )
    #         else:
    #             Z[ii,jj] = Zij(zi,
    #                            zj,
    #                            dz,
    #                            frequency=frequency,
    #                            background_conductivity=background_conductivity,
    #                            casing_conductivity=casing_conductivity,
    #                            outer_radius=outer_radius,
    #                            inner_radius=inner_radius
    #                           )
    G = form_gamma_casing(frequency=frequency,
                          background_conductivity=background_conductivity,
                          outer_radius=outer_radius,
                          inner_radius=inner_radius,
                          casing_length=casing_length,
                          num_segments=num_segments)
    return (np.identity(num_segments)+0j)/casing_conductivity - G

def HED_Ez(x,y,z,xp=0,yp=0,angle=0,conductivity=1,frequency=1,moment=1):
    '''
    z component of electric field due to a horizontal electric dipole at halfspace boundary
    x,y,z are location of observation, either as scalars or arrays
    xp,yp are location of dipole at surface
    angle is dipole direction in radians from x axis
    Derived from Hohmann and Ward
    Corroborated by Bannister and Dube, 1978, eq. 36
    Uses e^(iwt) time dependence
    '''
    k_squared = -1j*2*np.pi*frequency*mu0*conductivity
    r_squared = (x-xp)**2+(y-yp)**2+z**2
    kr_squared = k_squared*r_squared
    ikr = 1j*np.sqrt(kr_squared)
    r = np.sqrt(r_squared)
    # k = np.sqrt(k_squared)

    # distance and depth scaling
    ez = np.exp(-ikr)*z/r**5
    # horizontal distance in the direction of the dipole
    ez *= (x-xp)*np.cos(angle)+(y-yp)*np.sin(angle)
    # unitless term in parenthesis
    ez *= 3+3*ikr-kr_squared
    # scaling factor
    ez *= moment/2/np.pi/conductivity
    return ez

def HEB_Ez(x,y,z,xp1=0,xp2=1,yp1=0,yp2=0,current=1,conductivity=1,frequency=1):
    '''
    z component of electric field due to a horizontal electric bipole at halfspace boundary
    x,y,z are location of observation, either as scalars or arrays
    xp1,yp1,xp2,yp2 are locations of ends of bipole at surface
    Derived from Hohmann and Ward
    Corroborated by Wait, 1951 and Inan, 1985
    Uses e^(iwt) time dependence
    '''
    k_squared = -1j*2*np.pi*frequency*mu0*conductivity
    k = np.sqrt(k_squared)
    r1_squared = (x-xp1)**2+(y-yp1)**2+z**2
    r2_squared = (x-xp2)**2+(y-yp2)**2+z**2
    r1 = np.sqrt(r1_squared)
    r2 = np.sqrt(r2_squared)
    ikr1 = 1j*k*r1
    ikr2 = 1j*k*r2

    # r2
    ez = np.exp(-ikr2)/r2**3*(1+ikr2)
    # r1
    ez -= np.exp(-ikr1)/r1**3*(1+ikr1)
    # scaling factor
    ez *= current*z/2/np.pi/conductivity
    return ez

def VED_Ez(x,y,z,xp=0,yp=0,zp=0,conductivity=1,frequency=1,moment=1):
    '''
    z component of electric field due to a vertical electric dipole in a halfspace
    x,y,z are location of observation, either as scalars or arrays
    xp,yp,zp are locations of ends of bipole at surface
    Derived from Ward and Hohmann
    Uses e^(iwt) time dependence
    '''
    k_squared = -1j*2*np.pi*frequency*mu0*conductivity
    drho = (x-xp)**2+(y-yp)**2
    return moment*_VED_Ez(zp,z,drho,k_squared,conductivity)

def _VED_Ez(zp,z,drho,k_squared,conductivity):
    '''
    z component of electric field due to a vertical electric dipole in halfspace
    Mostly, helper function for VEB_Ez to avoid recomputing k)
    drho is horizontal distance from observation to dipole, as scalar or array
    z is vertical location of observation, either as scalar or array
    zp is vertical location of dipole
    Derived from Ward and Hohmann
    Uses e^(iwt) time dependence
    '''
    z_true = z-zp
    z_image = z+zp
    Ez_true = _VED_Ez_wholespace(drho,z_true,k_squared,conductivity)
    Ez_image = _VED_Ez_wholespace(drho,z_image,k_squared,conductivity)
    return Ez_true-Ez_image

def _VED_Ez_wholespace(drho,dz,k_squared,conductivity,moment=1):
    '''
    z component of electric field due to a vertical electric dipole in wholespace
    Mostly, helper function for VED_Ez (i.e. structured to avoid recomputing k)
    drho is horizontal distance from observation to dipole, as scalar or array
    dz is vertical location from observation to dipole, either as scalar or array
    k_squared = -i omega mu_0 sigma
    Derived from Ward and Hohmann
    Uses e^(iwt) time dependence
    '''
    r_squared = drho**2+dz**2
    kr_squared = k_squared*r_squared
    ikr = 1j*np.sqrt(kr_squared)
    r = np.sqrt(r_squared)

    # make sure ez is complex
    ez = 0j
    # get that stuff in the parentheses
    ez += kr_squared - ikr - 1
    ez += dz**2/r_squared*(-kr_squared+3*ikr+3)
    # distance and depth scaling
    ez *= np.exp(-ikr)/r**3
    # scaling factor
    ez *= moment/4/np.pi/conductivity
    return ez


def VEB_Ez(x,y,z,xp=0,yp=0,zp1=0,zp2=0,conductivity=1,frequency=1,current=1):
    '''
    z component of electric field due to a vertical electric bipole
    x,y,z are location of observation, either as scalars or arrays
    xp,yp,zp are location of dipole
    Derived from Hohmann and Ward
    Integrated using quadrature (scipy.integrate.quad)
    Uses e^(iwt) time dependence
    See Wait, 1952 for analytic integration in terms of 
    generalized cosine and sine integrals.
    '''
    from scipy.integrate import quad
    def complex_quad(func, a, b, **kwargs):
        def real_func(y,x,*rargs,**rkwargs):
            return np.real(func(y,x,*rargs,**rkwargs))
        def imag_func(y,x,*iargs,**ikwargs):
            return np.imag(func(y,x,*iargs,**ikwargs))
        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

    k_squared = -1j*2*np.pi*frequency*mu0*conductivity
    drho = (x-xp)**2+(y-yp)**2
    ez = complex_quad(_VED_Ez, zp1, zp2,
                      args=(z, drho, k_squared,conductivity),
                      epsabs=1e-20,
                      epsrel=1e-12,
                      limit=200)
    return current*ez

def form_b_analytic(wire_path_x,
                    wire_path_y,
                    wire_current = 1,
                    frequency=0.125,
                    background_conductivity=0.18,
                    casing_length=1365,
                    num_segments=280,
                    **kwargs):
    '''
    Form the RHS vector to solve for casing currents
    Uses e^(iwt) time dependence
    
    wire_path: A numpy array of x and y positions
        x in column 1, y in column 2
        origin at casing
        These are nodes, so there must be k+1 elements where k is # wire dipoles
    
    NOTE: the analytic solution for Ez only uses the grounding points
    '''
    dz = casing_length/num_segments
    zs = dz*(np.arange(num_segments)+0.5)
    xs = np.zeros(num_segments)
    ys = np.zeros(num_segments)
    return HEB_Ez(xs,ys,zs,
                  xp1=wire_path_x[0],
                  yp1=wire_path_y[0],
                  xp2=wire_path_x[-1],
                  yp2=wire_path_y[-1],
                  current=wire_current,
                  conductivity=background_conductivity,
                  frequency=frequency)


def form_b(*args,method='analytic',**kwargs):
    '''
    Wrapper,
    to be changed to point to different functions when desired
    '''
    if method=='analytic':
        return form_b_analytic(*args,**kwargs)
    else:
        raise ValueError('method '+method+' not recognized')

