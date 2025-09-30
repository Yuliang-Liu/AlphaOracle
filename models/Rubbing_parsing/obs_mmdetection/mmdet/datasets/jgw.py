# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class JgwCocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes': (
    'sentence', 'p8w7ujqanz', '6kf650i4zk', '963x9919ok', 'n4g5hrawu4', 'mg6ldxzwwb', 'vg3ptni3mq', '6wyn6yddb6',
    'fivzk82fdh', 'jnpp6r84xp', '0wfbo7ml5k', 'no4gdlajf6', '1axafi8r7u', '3kadgutgs9', 'dvdjluqxnp', 'u13qoz09bs',
    'vbkc7p8qc9', 'zmixmtzym6', 'nak4ioxpg1', '8jtrxi583p', 'ftj0qy2f7b', 'cff266uh7y', 'auprttukg2', 'egsgsijlte',
    '7udfap69mj', 'r4xlmwmdvl', 'i1n9hhpy74', 'd6k4isufpw', 'owcp73s8go', 'urq1zuk7jt', 'eqo29r6t59', 'rzfokwkc7a',
    'ptr3rfuph8', 'jbgbxt897u', '0ppa3hvekl', 'pkibwuo4bv', 'vk0iw3x90b', 'o63zqxohsa', 'hjfwaxkb60', 'ow66z51pa4',
    'crd6njmjax', '1ttzl6vlzs', 'nscqmvthk1', 't4tioxlhov', 'wk8kgtqz68', '1fvbbmvepl', '2x2pp2iwbx', 'd20ybazw7f',
    'lpjmcjuz4z', 'aawiifil1h', '2kw4aito68', 'w3kng3v1nh', 'e1wrdxp259', 'fhr5fzfhst', 'ya2h4cjfaz', 'mw7sb9ey49',
    '4a6130bz6w', 'vaw8wrp6mi', '2kvlcqk9xy', 't1r1jjd8v1', 'ivbw1o50yg', '67xwlxwxwm', '0r2vse8o96', 'sq81wh8r3p',
    'z2y1vsfy32', 'r0gb9nvxhz', 'xjnn5c8ctk', 'ti8ubql3my', 'x8dr6vkpa0', 'du1603bntk', '8tu8ldh995', 'fy8197yoy0',
    'xhfae21bpk', 'xh47ul1r1k', '1cqtj1q73a', 'iho3azmdgu', 'ctcra21kbu', 'rneab7qf7c', '5yg8yh62tk', 'qmve1owxof',
    '08fa78pdp8', 'm87lblai6p', 'yvbokzhzmz', 'o080wcrzq5', 'j5q9a8d49b', '8xhrt6fscw', '92dfbxlxeh', 'dtu9bl4faf',
    'shksxlt8ok', 'yfnx5cwgq6', '7nuzvu1uoz', '7tszx5wu4p', '944ledw1r7', 'o6522xq6sh', 'xwpx2aiwi7', 'pfscmtnqbm',
    'msctk8rvba', 'vudruhunxi', 'ihzje59ylg', 'x9y4slwu2b', 'gfyngwaipo', 'lxao6q3l56', 'l62kyktzq2', '3k23i3cwyt',
    'el7xh0v2ov', 'hed7mevt31', 'lebb92n5gt', '82z4h91v4r', 'usnmo2t72b', '4n6v7cj6m9', 'h4ivjjlrr8', '3wku1ohk1p',
    'j904v6fm79', 'o764sfprik', 'h2mhpxskhk', 'c21flybnax', '2h3rwms23l', 'jbo9t31zqp', 'k6l649jf98', '1mu6nprj45',
    'do72nuhnr4', 'j29igjvgwl', 'r09trgtxjq', 'u5upr132uu', 'h44zkm9mt8', 'e5kvpngkc8', 'pa13rewdd9', 'z5begenlrf',
    'diesgffg1o', 'fn6hszube7', '8uh1kfvlhb', 'xwck9dcioj', '812nh3gzrb', '7z5dh17d4b', '35mlsnk38c', 'g6i8yc3t6f',
    'e4cn4k8l46', '1zq2hlx01l', 'tqko7h2wrh', 'wko0jz3cod', 'aqf754j3rb', 'k7s5hlaoz3', '2i3zpwl5a9', 'b6f7lj4wbb',
    'id0m0j1kjy', '414nuewqwd', 'dowi3lt0dv', 'kcbhwd2hb6', 'msl6b81ikx', '991ix0j4pd', 'lyugdv49gk', '49lxhylgu1',
    'hl0ffm8toi', '8i22vg823w', 'bbf56n4pg9', 'gxqbf8g1vc', 'vrcmv7tjdu', '6b16j83isz', 'l3wbyhysiw', 'om61e8d11r',
    'km215tqui1', '1g215frscw', '44ocjtdxxw', 'hoshv8o78k', 'ksurlnymq4', 'ultq017ef8', 'g9e7qyy71p', '1b3rdlleou',
    'dvqszc5g50', '9tpoluos1q', 'xkubtjk815', 'g8k12dfmg6', '01v0id7th0', 'zc2s5bv3x2', 'rlx0knkq0v', 'hxfd7qqttz',
    'rkegfad4bl', 'zxuw992m8p', '1wismy95hz', 'qzgadtxm9i', '6vao0zrbrv', 'aki498y6jc', '66iowpdbuc', 'y1fizk7ybg',
    '3uiwedg1ki', 'h204u5h9ms', 'nllrzldww6', '0r6eikzw8w', '4nk5jzlf6m', 'aoaipuy20h', 'ajjda965yb', 'wk0l68dlkg',
    'vitvodung6', '51laoswrwb', 'wq47pyztal', 'nms96pmn1w', 'z5hsbrw198', 'n5t0jhw1aa', '9igytxwwzh', 'fzv0acj6pa',
    'h920pdxci1', 'gbjr7sabn4', '1ie27xm4md', 'jvuf4ut3c5', 'fi0wiw1adi', '70yhyiaqrl', '34ks61j1m8', 'gbcz1nzomu',
    'bcsv0dsf6i', 'bayjrx4i6e', 'mhzqyq008z', '5tvchkyhcj', 'hyuma85xcu', 'j6kh0jkptk', 'kz4u69knjf', 'wx71xtznl1',
    'cvby5rww34', 'h9hoshfwwj', 'co7kntv3lh', 'm6y8bftirt', 'cky14hc1l9', 'cdalwkdxte', 'lkbv9me4ks', '7or1k4syoy',
    'cyagulc7ha', 'lv9rlaeib9', '4krd3qcngq', '4pz6izsjbm', 'lsjuogr2ev', 'jstya23ki5', 'xuf3yby22w', '7fgd3fbmzi',
    'fmv0hxp7i5', '3p906vor19', 'p1t11rbor2', 'awg8lyvedy', 'hoz4ouh5tu', 'rnar1q86c5', 'kn438rbkpd', 'oe7t1db7i8',
    '7b9lfpes9d', 'at88lnryep', 'er8jkra4dx', 'w308vycuhp', '33ico48nyd', 'fpq0j1p9ii', 'b4g0n16lxd', 'cxzdczacxy',
    'qj5ls1wkvt', 'no8hgfw0su', 'ge4xfsbtgz', 'ls6u8dscwp', 'd339lim5ps', '91wuvmq6ju', 'exg6lsn5w7', '0zij73e4bo',
    '933ie56sph', 'w0kil4h1qz', 'wf1lpsb2vj', 'h9rt4lxd99', '54s1u855x9', 'xyp0ngvyd7', '2bcusrt88g', 'utkbozj9sh',
    'ewiqwj2bq8', 'xdimpzk5q6', 'adohx76jpl', 'krjxuj8jnv', '18jj79so8p', '3stulpqfe9', 'ig9oetqcmy', 'nqghhf5br6',
    'eakt7tagpp', 'ed04il53n7', 'xgtqk1w4ya', '792pre0fm3', 'e7simxzm79', 'nnkv7fnqj3', 'k9fz66biv5', 'y0f5gu3q0a',
    'chdjguobk1', 'dgw10rc4gi', '5zjto89nh6', '5t6au2olk9', '1vtfwm0co9', 'czt8eczekd', 'apuapt9wig', '4gy191xe86',
    '8gro2b24ja', 'hr21gpaogg', '0k2px01a8e', '90a3utk2wf', 'khg8b4r65l', 'ah10xtkbxp', '1r0owtspdr', 'qpe695ed6v',
    'x0wt5y5bs2', '7kjgfmjunm', 'vp4cxkje4s', '32rwmop1h1', '3xbmgmm4j0', 'kfdhplatyr', 'papxu0m6p6', 'bp8az8iysr',
    '4nvomvnfj0', 'ecbetcendb', 'dhbxzk9tyc', '30vdsr9swu', 'za6gsc8sdo', 'xa8nk3i0as', 'uw9w7g847l', 'xl187q597e',
    't2memeu0c4', 'dupksehej2', 'gwq1i0smza', 'lzfy8o1b7b', 'x5din0lbep', '6i3mdu0x5o', 'g3etmch2tx', '8638v39hf2',
    'xoxgvczyyd', 'ug07bm2ckw', 'ev5rkbdabc', 'voe2rlzy41', 'nq3neh9i5i', 'lezatbpl1f', 'gx8z45578c', 'ajmnqxfrau',
    '0ln5rvbsug', 'e5v2cd930z', 'm7l429tjt8', 'm32ngscsu1', '0fig5rmiox', 'orpiitt449', 'dfe38kc2ai', 'u6dc7i5is6',
    '446qjw4xgi', 'g5lujnrrvr', 'x4hv21dl61', 'rgq9fk24zd', 'ghaj284ctb', '76ba1nqq6s', 'dzob6ag7n1', '0guqpq6re9',
    'nonh2c9f7j', 'gig7gnjtk5', 'ixe8sbhdfm', 'l1rav4fxk8', 'bey28jrhwu', '4qrbm22mmt', 'h4ousqq8av', 'ptmpax2kn8',
    'u6h3c0iftd', '9zn0g0wd9v', 'krjznp1i0d', 'vzb2hyntyq', 'kzzehs3b2j', 'mkl3fx9uth', 'v8blfuvx4g', 'hlxb5g91kc',
    'bwcuiuwtaz', 'hbltymdyef', 'ssgmq3k51v', '22lvvv6gxs', 'zd8unl3rqn', '0rv771drzl', 'cdwlhoe7sb', 'l170p06wip',
    'e19hd5tm0i', '1ct6qvksrf', 'fpfxpu6u4z', '2fx5viusdv', 'm3gylabt6p', 'dmiddz5ehl', '4edy4f0uol', '846cza5dwl',
    'ptbmxdfm3i', 'jh1gwh4n7f', '8k76kao4ll', '804dv56eji', '8i9yf0yr9w', 'pjyj6rvuj6', '0umdgqu77x', 's9y9vbblxa',
    'ptd0rmmnrv', 'l3wc3uouh6', 'z403ojus93', 'gtckp2vvj4', 'zapprwvtx9', 'hq2zgeaqqf', 'x0w8luxhhs', '61p261idyu',
    'hwj18cvm0i', 'f8ilekqrfc', 'e165xleaqt', 'g7a146vwlb', 'c5vr6lndrx', 'fp0gk2d44n', '2lep30tiiz', 'i44zmo51c7',
    'f7hrrl0bh9', '6vdibqvqd8', 'xsfvig8s71', '9smcf52kd1', 'c0j4lyz6aj', '3lrj040600', 'iexlgj4cvt', '0cyukofrr8',
    'ih7qa28rb3', 'e33qe36es7', 'qh2w0s40kw', 'yz361h0hzf', 'c13j4nmoij', 'p8shoneevj', '6m1qvxsh3e', 'qxiorvfxji',
    'noyypvbg20', 'pc30r8fiiw', 'owlfrj4v76', '84umwz5kds', '9utswitc0o', 'dmt62prazn', 'qhmhxv0zyb', '60kr6bp9hf',
    'sst8wbq7kw', 'p5mp3xbn99', '8w1l6u1565', '243s3vaa73', 'xmfpjivigx', 'my9kducfis', 'umwt5oqyfy', 'yy93ubfmig',
    'uzw48xncgx', 's8bffjoah6', '2s8i3l7oy7', 'hxbszp3ept', 'uulphaak6s', '0eoslpndnk', 'i1qbimwfk6', 'mw1n2p1voy',
    'cyv9807bcj', 'mepmffeebh', '7azc4262oo', 'fhaame1ny7', 'dbubnsu1tc', 'ootfcayp1g', 'p6uj3n0aze', 'i3lktxnh52',
    '1yymlqlzl6', 'ayjlria39o', 'fd2xhqwsjk', 'd15m8dd17h', 'um3hp76w21', 's73v9sbqaz', 'ghyo99f0x8', '5zeokzncln',
    'lr27sngh2z', 'omtx3jpdtq', '5l2g3imogi', 'tpx3x8k9lm', 'g5n5rqvdxo', '1bdnhihf1z', 'bjuu7rps2w', 'rvr9wfrhx5',
    'lv504kyqxb', 'tyn2yfryvp', 'fg2clh0hsl', 'i72tgmcydj', '0fjs8401bp', 'ce7ek63y0o', 'vvqlhyb7jr', '9h2kq1x7jg',
    'rc6s5fhd2k', 'epvptolw9r', '2npqcwn2oc', 'vxs7geukz8', 'akl0kdke8r', '5n30uigzj6', 'n0bfvspr0d', '56uvvmej7x',
    'ep7qnow4pv', 'blujwexum1', 'a675ctdqop', 'uho2ds1twg', 'h81pl2i5a6', '27zri73hga', 'gaf7rmekoj', 'ly975u80wd',
    'uxd7qgx60f', '7nw813njs0', 'nufbzasiq7', 'dpcoqc8hoy', 'ohovbcjeax', 'imnwy7c15d', 'ow05gzmj82', 'lv32d9tz8g',
    'bh3e3oql61', 'ju36m47psc', '3b0uqjbbv5', 'gu3h3yqtk3', 'yrxflndwvy', 'llttpz4l8h', 'zsmvdq869o', '6p9s6p6wyb',
    'nxavm65mg3', '2tq601zel0', '4j5hjsdkwc', 'czrsscjxwh', 'cawhttc4j5', 'eadbesllam', 'avcjspsyib', '02fjn8nulq',
    'aetgi1oyaa', 'jshaxuelhn', '2aqd8x0a8p', 'u54u515u56', '784pq6tqo8', 'f3i76evvjb', '2ekj7pe4h8', 'we92o7vlu9',
    '4zhjejk00h', 'g7m9i371p4', 'nnm6jhv5jz', 'wrd3sifbau', 'r2tbnngvcz', 'm4rdm8ckxg', '4wsrhqopug', '9xlu134oo9',
    'tavwtgnfdp', '13cqabljgi', 'wl3zq7r5tr', 'e5bj1m9m2e', 'x4t1t00zpg', 'xvt1om20u0', 'sxmr1dejq2', 'tzqeya1t3a',
    'jmlf3qvty5', 'z6mdb6y8rn', 'x8df98b9m0', 'yokl5p9l41', 'ilh8kgxttw', 'ji97td3npi', '1k9ogi7w79', 'uzped615x4',
    'd8jnefb3d4', 'gcl8bbz6jf', 'svrqsl8ydm', 'qeqvpjsnyn', 'xhh4txsh2g', 'dsbrwro9x2', 'h24amfz7mm', '5js6z3yvqi',
    'os0bklwwmx', 'gtysuhhw3p', 'o6m37o3l7o', 'zh4yjr25wv', 'bwfz1jvlck', 'cvzszyp6bm', 'qsw4oc8hfv', 'rl4441yfop',
    '5xa8w1z0p0', 'rrua1697nb', '5xld7x764s', 'a7al7kr0c5', '5jkdcdmi6k', 'ehcz1jiv3d', '1m93z2t8wf', '1u3e5w4k2u',
    '306br54upg', 'lacjgjmulx', '947jc8d4sh', 'rfxejqz8as', 'sbjf4foazl', 'sdos1cdyy4', '7p7pi5fxkj', 'd9vtt5msc1',
    'yzii0m98zy', 'csyhzui9uc', 'kjeovtb3db', '60lzmbmrwp', 'x4mvyzj0ob', 'bw6bl0elx5', 'tcvw6ggyds', 'oz65ogq74g',
    'j2m24wzu8g', 'fzkl7e4vel', '00bv3k93fk', '8ppcj3wjwt', 'a73fq76370', 'oaaqvh5dtf', '4j4r3esjz6', 'jwdfza01vq',
    'hrejm00eyu', 'dccq423sp2', 'z9zzf3vavg', '4jtkq6w8f6', 'czou0mg13m', 'rfc7w6n7w1', 'bck0nedio0', 'gnt13du9z8',
    'fmueu74w0p', 's5w45l8o2m', 'up9b7hui1h', 'ugy1kvy5yz', '107xndjbdn', 'z052z9k6c1', 'g546qlgz9n', 'xblum49kus',
    'zgipta73eu', 'npgbktbhuc', 'tdznjmpywo', 'l79u9elpco', '3tsrz44re1', '901gohvcyr', 'wy7qlm9hjx', '20tmlpf5rq',
    'vl6bxvyj5s', '030q5fqlrh', '8amt82vs7o', '7xgwsikqad', 'x3zcurblzm', 'ekzq6rea24', 'yhwcwu1uot', 'd7mczw6osp',
    'j5cl5jcqdz', 'r79attbhro', '43h8h9jdqj', '9iaq2j5p0w', 'gsp1aliedr', 'bg8bhr1896', 'ce6hpt5ju5', 'wex7jm17pb',
    'valpak04h2', 'qo76izuv9q', 'qhpxus7k8t', 'tp2kuz956n', 'cl4srojc09', 'oo1wr8e24r', 'q8n2xmqs1o', 'q71z8tnnc6',
    'lbm101454u', 'zsc0qawq79', '7wrt6jfhfq', 'jupa4d7mu6', 'yzl6vltidf', '7jjbgcqg1h', 'sjo5wujrih', '14byfgv30p',
    'xnaggs7qe3', 'szqyf48ni2', '9aai46wd9w', '47ioadp9wd', 'x40w6mop83', 'urzeocieq8', 'kyjb7o7olz', '50z922qja8',
    'altlfztjg5', '188qgh86vq', 'p7rq0a1geq', 'zjs0019zm2', 'z8s6u45g2k', '5uwqhmpbpw', 'npl316l4j0', 'refqnby49o',
    'fbmole6hwm', 'mxjh9h64im', 'r9cvuw9v1t', 'vgtvawlq60', 'umalf001uf', 'sevo4xa26q', '9ayc3k0dxu', 'rxvyya07z2',
    'vaz8c6nubx', 'r89cx7sv2l', 'a9n7vw34kp', 'ux1lvrjiu2', '084xuar347', 'jjzvqk37cu', 'vqo5659li5', 'dnmwbqd36a',
    'ntmjixw7s5', 'jcsf6ohzo1', 'dagr43m1aj', 'qbwv11qfx4', 'alvcqy8hjj', 'm4ruun8m3a', '32pzmjlrd9', 'hvm3m8ypgu',
    'y7e97sd82a', 'o7rxqft52e', 'if2xeoq16w', '9c9zf6hv2z', 'rddtpyuik3', 'a7fntjnau2', 'yncuso5ryt', 'nxv16lsfup',
    'rzyrjgiv6m', '0cwxw7rveh', 'avzjpgqkym', '8kl4colp8e', '2bw77slbb4', 'jiu9nnsqms', 'kzmks2fxlw', '46aok4e9hb',
    '4ytix1aozf', '9xhq4zclpe', 'f1uojpw9ej', 'fmdym8mxmv', 'mrz7t6gxh0', 't1wd9fiznl', '5cuto8p2fx', '6bzmgp7e84',
    'b037aq4o64', '0kcfgksim7', 'xlscyddubk', 'r424gd7pwf', 'awebxd6wg1', 'a8yk8tkxc4', 'ho95nnufi7', 'ketncivolx',
    'bor1nhz1d4', 'doqvaj2vzu', 'yhqkhecxxc', '59lmys7y2s', '52t3nzmp0l', 'krhtv9twvr', 'qom5qq71bd', 'b6y3pr9x2a',
    'n52ssmqnda', 'fgo9aomfqj', 'mvl9bio8ou', 'vszyfvmzzg', '9649n5rjvr', 'rdt31ynd6t', 'jthh3lgecc', 'rysjgchsua',
    'iscjlutw9b', 'qgiwl47lvm', 'usqcyemfku', 'l2vxxrjn36', 'ze9pkdkqjd', 'jn8qw1nhns', 'vwahtur1cj', 'c40r3zyq5t',
    'c57ck009xt', '0u6c7rk2ux', 'tpy5799qxk', '8pu7n71zyl', 'cun5nm7d0f', 'khqn7h3zps', 'xbrbbm12wm', 'i5iv3migqj',
    'dc22oah1v8', '6k7x6eaq4d', 'izzmxjxqh6', 'hhr47shesw', '27jl2n6uoo', 'xjoc12tti2', '4e3li7uc27', 'uwjljg3tt4',
    '42b1o4c46w', 'zk3pz24x5s', 'wxpi4rjz2p', 'wu9j4fyp5p', '1vr9xpsws2', 'kbac52iw6u', 'piqtpz29xl', '2nz3zp3pmu',
    'dmio574hpd', '7fhnqn7giq', 'x9q4jluo73', 'cit05izhs6', '59km3f155u', 'kgfsb68wnb', '9mbfk67sq4', 'hcimcis80q',
    'xdjmj3ywdf', 'p1fm9nse78', 'a4a454odk7', '4vpgmxwaa2', 'upv9mf4trj', 'mymkzhalht', '5u17bmbs6p', 'kz1rfdj0ws',
    '5wxvkutef7', '04jjwtizii', '9la8v54yz4', '50oggjbzcs', 'lv4vlfcnzr', '7y2udd7ppm', '10sh1f47vw', 'm9q3n9zj10',
    '1l0us6kkug', 'jgdbp661hw', 'shhzugl17n', 'ckhg2g5iq2', 'lsh0azwee5', 'cw5ssmu5j4', '0xcwx8sf6z', 'z4bhaqrpt0',
    'myoqtwlezn', '9y6qgnclpk', 'jl0fjl7uo6', '4tyq3wzq4i', 'dm6pg50uwu', 'tore9qbutp', 'u6bkwnepiv', 'leg5b7etfr',
    'su2e76dtfo', 'u7j8pvl5zq', 'epuiyzcv4m', 'qn4p4ey9d4', 'kdx7k2ou1m', 'ld0qrctsbb', 'cdwgyw3kul', 'ciduvuy3ar',
    'ohj3rkyczg', 'sx7ig5dria', '41zmp4zh5e', 'ch3eogm319', '6dnfyfs9ol', 'otbtv0ewaj', '2rptsk0t7g', 'd9r8vouxue',
    'w8l90of6xz', 'gjgujysmue', '74alaf1szs', 'ctmnbdx9k0', 'l12a906yri', 'sixop241yv', 'oi9nd1nd9n', '9p1cvclas4',
    'hvoexje4ex', '03a29zpxqw', 'ropjrw8fc8', 'frmptf4le7', 'zu0urjl5fr', 'si8qqdmshi', '4yk9wrn272', 'gwuthswkc9',
    'q40isurmbx', 'rbnqxy5oij', '31gq0i3w4d', 'zkrxof2sy9', 'b9q6ewwhki', 'vtg60f34gv', 'bxk4wfwnci', 'h8exoqnpru',
    'm6szz2tcxv', 'hcsv6oxj1l', '2sdz63p9g6', 'hunv1ff3gy', 'ntecsn5xqw', 'lcxtisdp6y', '6tjo4iag3l', 'brklsll1zc',
    '77e3f8jri8', 'd0hauaf16v', 'fdqh7fcjf2', 'yqwtqscvwf', 'tr7pel9ndw', 'ogt487t2u1', '4y18ww0a1m', 'ls1cvnh96s',
    'kmbtnosiha', 'n2eg801rgy', 'earnw14zid', 'oe6xmwn3s6', 'zhdr0vtntz', 'cpr9kqrk9x', 'g5rxhh05vn', 'k488tbxgjl',
    'aq5aoy9xbp', '16xzhmtg43', '5d0kwz72yy', '0avi2tcy0g', 'ar259dtulv', 'f9f0lfjetk', '1yxyj1hxvi', '1ooijljqrv',
    'wnk1om8h9s', 'wh11q5vypx', 'o8jmn7wu5e', 'hm44o8jbph', 'u4u121uh4d', 'b1ojfpwayk', 'tsa3qlzed1', 'tp5ykv5jzh',
    '0bcpbg0geh', '5i4ba39u9k', '79c9fojxw0', '6p6efwrmid', 'f5d3yn2so6', '73ru6libmk', 'zreu243tyr', 'c7qvt9bwt5',
    'ywoquzurvl', '6nn3a042hn', 'otyf9c85b1', 'i2m9i5ytbg', 'hzglrwxdtt', 'y5dftmxvj0', 'ff8rp0nh6u', '9izvvjw1bu',
    'q3hrv2vesz', 'nptkahm6a5', 'k2sr7n4ffj', 'bnzum9xkvt', 'ydpnp91kss', '5xzg937unb', 'v34viqk9rx', 'r1lyc5x0pa',
    'fkhhsjy5jk', 'v2nizai7hr', 'dhewuxyotl', 'v0vpglxpcz', 'n9qxfnupux', 'vgvqkrxgz3', 'kgnlcsowqg', 'zgocda0mov',
    'ng1qgo1hp7', 'oeui2r7lqj', 'irwwgkc3py', 'yjzpdsoyyo', '0y6gjyc5xl', '0gmn9tzmlv', 'q1t9301odh', 'o3uagweguf',
    '5n56p5znf0', '97dgghy3oj', 'y7ndywj3w5', '51p2xxpvps', 'zi2pxt9kk6', 'acx6uyop18', '075wnah299', '9n8ef83n5l',
    'buakj9jija', 'luw8r1fj9l', '2u8lblv476', '7jecspy8fk', 'o2g1mpwg2a', 'ake9481rl7', 'fu6caa5q2a', 'gqezg03aqo',
    'fhseyjwogt', 'dlpctjqwqf', 'bsmx94pct9', '3lizt17yhy', '0wmcj3ydi8', 'wwpyiw1vm6', 'h521w0m697', 'cz9doc0d73',
    '1uxh6sccbd', '34yhwco17n', 'qtwjt6h6gs', 'cutrxw7e48', 'ae9cjv8zji', '8ggcq3obcu', 'qmsdu51da8', 'j7dwqskzg0',
    '5k1l85d8n2', '8g4h86hnc3', '1vs1krahfs', '353gxm6hd4', '8st8h2b5yr', 'dd5w09rr1q', '4r0g6bx5rc', 'w4frb60x9v',
    'a5tuvlp0po', '8u1hzbiq2w', 'snlsmzwecl', 'ekqpz9e16e', '5p2qekgax3', 'o68jkkro3f', 'nr7t4q9h8d', 'nv9w78focj',
    '25iafmky1x', 'm74wrlidvc', 'riyuog5dwt', 'v6sm32hcto', 'hclsl73wg4', '0qc20cajlp', '237k640lqx', 'aphcoumla0',
    'ck9d66h56b', '58rt2jv8ut', 'nwi5rk9lkb', '057g0jgjld', 'aelrb19u0i', 'z8n1dxtc9q', '9tdmm4guev', '1rnh5muox4',
    '8t2hy7hbrc', '9vm0d7txqd', 'viv127p4q5', '6hyn7a9bnx', 'cdd4p6xm9e', 'bu3h266vxh', 'z640bkv6r7', 'f0j9ho8ua2',
    '98ipxc1p42', 'u5pens8hl2', 'xcu1fsj3pe', 'e1p1kzz96u', 'aaw1t17vqv', 'yg0sm41ear', 'ex96ntmoie', 'nmmmkasm1v',
    'aoxhv8jrax', 'a8n7ywfrd3', 'phjj09b89l', 'ggmxiphq64', 'clwxtgnjpw', '16p070gidp', 'nu6yj5ke73', 'u878nsmv98',
    'ihsuhqgmdc', '8elvo4ws82', 'x5uz58e5p9', 'ctejf56of8', '811h3kcxvb', 'akt7yld9xk', 'wp0tnl2ydu', 'pfzarn59ab',
    'iftkudcar6', 'm2rpfrft4m', '6wyksx413f', 'wujt49n5q0', 'axnfyyroo8', 'kei9slmf3p', 'yo5h3qdl2s', '8fkudp0t2j',
    'ywtdw3ws46', 'lyy6c5kh8d', 'j6pe4lu59v', 'hxwcv9yfyk', 'xq6hizhbmv', 'xqstwjkti9', '5fsjukjyjt', 'egd60be7r0',
    'bejesjhtdn', 'dxvqlu3dcd', '7e0bc7s6s0', 'bg0evhjvox', 'zolc6o82zt', 'kbn6bnvddu', 'g3hpga6qu2', 'cbi1n9904r',
    '38kvsr6xn2', '3gvgmurmcb', 'ymr2raryhz', 'ha4qh8t2sw', 'rv7b9x20z9', '5t3mdoiyy6', 'z0pbcna9kk', 'wik93edkn9',
    'bi8t9sxwtl', 'qg6bx6xlgq', 'kzmitqixsp', 'kdo5ozx6cw', 'ahpipvmufk', 'ojtpg03v54', 'xe5j0rm6ao', 'r26j6lro1i',
    '81k3ssygt1', 'g0o67a8jpy', 'xowmsv7xvp', '1w98cgnanq', 'ix033niy64', 'efgs491edx', 'uhz6qrd2d9', 'cixqnyxp9v',
    'a0xx5u34nm', 'dljr7e3ubv', 'g8fzkhzlvg', '88n7gfxlqz', 'pzuji4u0at', '587vsj71cs', 'kk9zxal8u7', 'w06mzbrv69',
    'sawqk4xfna', '3eorzalgl4', 'r90d2o0i9y', 'ks7w8e60q3', '0vi83yha8h', 'ebehpv21xt', 'i72bd4i3o5', 'w4mg32kjp4',
    's1n3ksksbg', 'd7b90j3d04', '94cjz9jg9y', 'ny2e8bby33', 'i4jbxr3gd4', 'pmwfonj2ag', 'vgdnx3zrkn', 'du93rs9pec',
    'u2bx2sw1bv', 'fyihpr2f02', '9mg6ifqmny', 'fppjmg7fhh', 'tnwvrq73qx', 'kkt3pdctdg', 'xeawj1t30j', 'l7cyvm7kbt',
    'iuzkqe17tf', 'fybnj2savm', 'alc6eyehym', '9oqm32g594', '57qxiq8tsb', '9nd2ajl4nx', '7wp6czwrw0', '8sxuwvgo44',
    '2sxbksrwvb', 'ea22soll93', '0h1vf0twxr', 'qtih3ae3j0', '94zt6dsmdc', 'n3yt7xt78d', 'hu47oaqoqg', 'cr5nhxdhbt',
    'oho66symg6', 'xwmrgwb38v', 'gyrhryqj34', '2e25hd29uu', 'ytvfk53ys4', '04jb55nlsh', 'soaedltvjs', '8sbbzthl9w',
    'zxrxdzrytu', 'pzvzykmf5e', 'upw1kn3nga', '1d6q9ds6rm', '7l5l1f4dh8', '86d04i30jc', 'h1yvqbcpot', 'qdft7im9cb',
    'fcooblbqyz', 'zz9f01s39q', '5t9tgx32x2', 'n92bww8sym', 'hcj1lu3l9a', 'yomncuf13y', 'vf4c383b64', '7nnyrveaif',
    'tvnicmgqzf', '83rbcs41fn', 'z78hrd26su', 'wxej0p8z4b', '3cv4e1htww', '7w3mprfyzd', 'fcmdhs3vae', '9uixxn6prx',
    'srr795shtm', '6w6nubiwsc', 'm6guxwjazt', 'qmspt85dhp', 'bzimognfvp', 'j3rxkj56cs', '7mtakgmbba', 'sywfwn7sg1',
    '7h3wu2xyyf', '6ceuhy4fvr', 'twk4rdk9gc', 'lp8x647qo2', 'gx21ndp7yy', 'r03zb70e9r', 'mnuob0fm04', 'zn65ejrc8f',
    '0n73hvtfpr', 'eq335my2ai', '7ls5j1vbpq', 'mgs4xhu5g3', '33r901db0e', 'reqn09rgft', 'nmwnjd5u4n', 'bfwtwd92m1',
    '7es6m890l4', 'akwyl4gzy2', 'dciu80t6yw', 'znikm1880t', 'giz4x476ug', '7q3ryyen2t', 'lfox6x1ojm', 'pmjqw82db0',
    '2ao64scs41', 'nf3yu2ct27', 'd0mw2u39lg', 'wfyy86f1n0', 'xclyxu4uku', '28yjralexi', 'cvd6s2s17g', '7f1o1pqd9w',
    'pupkl4x8x3', 'zog7b0d83z', 'bcu00f5y6s', '5diuh9hkw7', '5wa7dg1020', 'c771nmkmb8', 'dc1wt3fwrg', 'h16yr7furr',
    'vha0ssegay', 'upwujhrby2', 'umabbd0jwi', 'oav68vru6u', '0bwtsvgzvy', 'wf3aejbrt3', 'drf1i3fmxo', 'r47d5160c7',
    'oxa0ylq9ay', 'af435iu76k', 'swkogajfub', '47v5slu9p5', 'mywhauy2tm', 'mc2tdm102y', '9w7u4m625q', '6k62imdisa',
    '52a130pcmy', 'un7wnrhutk', 'm4dq2q9jiv', 'mm49eilaqb', 'giysoy67zo', 'hku4d0ck4n', '41cgu5ewmd', '49xqdx7w56',
    'ncgekxox1o', '2bcwadfq0r', 'dxasp1shh1', 'jyskjcj861', 'a4weam3zks', 'nuj3f1ea1d', 'r1qimk9d4z', '3ade45dwte',
    'knps4jrxz3', 'eq0zcoihok', '9vw70byryn', 'h3nw474wwr', '81cqeht6m8', 'h82z1bo3i3', 'aotzf9i4t4', '1vataolxj9',
    '32g5flgua0', 'wg598fdf4n', 'dhqar7cnn9', 'hmne5uy3qt', 'yd0p39g0ly', 'uhk4plz1rm', 'brlxfsqi3r', 'qe26dg343r',
    'kfb7sj7i0h', 'l9s5pvc4ia', 'lyq3tnc036', 'w49pivmpx3', 'qtpfktwrdi', 'xbypbgiy9z', 'qmvfvw99v9', 'xllb905eed',
    'relqgnqija', 'hmxcbd3z22', 'xh73qi524y', '67yxx2eqp0', 'l4lq5esn5g', 'zk0j4pcnzj', 'gs24si06qj', '7hsrnqwx2f',
    'tbecf7rhkr', '9ypw4dceci', 'y1u4nexjl4', 'wexkkj22qu', 'mqs7u6rngr', 'qyg9g9hii5', '90zmojjf7j', '4s3tpt5d8v',
    'xeyf2mkerf', 'pp2w04l9tf', 'sqh9trpulp', '4noufqnl4f', 'xcwma5914e', 'v7v6ucaw72', 'sj4e2q73so', 'yunhitgxgx',
    '4nm99xtfab', 'iuouekyjwj', 'ls6kprxw1j', 'f31a6zuqwf', 'zr3h63w2fy', 'nclwec7ns3', 'jszwtrqlai', '0tk0pmitom',
    'pmx01iziy9', 'p6gn06lpie', '2p7v2fpe0b', 'egf0h3s4xh', 's4dnc91sl4', '5sgode9pne', 'f7hr5ku09c', 'foufn9pt88',
    '65l1kwga5y', 'u7v6rlhp81', 'bs962k3l0i', 'cffpoun53u', 'wuqq17zjcd', '1tvpevfzmn', 'ktmu5fowob', '03w9kgybpz',
    '0lplfrdm68', 'yi7vhns7j8', 't997jxohr3', 'ghln7uplih', 'n1l9ttl9ei', '7cerpwngv3', 'w7adb17snu', 'ify8wp45mz',
    'f1vyjgg9t6', 'jxtkszwkrq', 'srhk7ugxm8', 'hftl4m18q0', '7gvythc7mi', 'yuucprnzb5', 'mqfyrz6sen', 'dlza5ir0k7',
    'o7eb365hnf', '1rzrhj2j4j', 'yszlaw22lm', 'umntls2dlq', 'l93tak6vzx', '6fnrc025mn', 'rfhhuaqrts', '3drbkkpjwi',
    'gtjlzc5fyr', '6ewkk2w7r1', '6x8v1c8903', 'k0pxsd02si', 'c4uzt5w1qm', 'cv483h1fw2', 'c3d4dbun3t', 'u3lh7tekno',
    'medazzhc49', 'q4rk3syf2x', '7ezpstbc1y', '9xco9z4c3z', 'e8qibnliym', 'hk9f0607uj', 'a5yunoxatt', 'p4a5rqs03k',
    'kyjr61sdkq', 'tbh9vqe3ev', '40g1faed45', 'n3oe6amcoo', 'a7mak4wuom', 'n4tua57hu9', '8p0clmx851', 'phxqocglhv',
    'anku02zxgu', 'fxgbagkc2t', 'vpcd7inkf9', 'cvxsbiqy4g', '55qv0winah', 'tq26tb9tj9', 'dc2yeai9wr', '2hdoxmv8n8',
    'tralay3z26', 'puh9mctza1', 'emizdscukm', 'txqs43zo24', 'tjbyfvpm84', '7eni93zlhs', '30u8rx4qre', 'gc9i9qy9cc',
    'h32m5ulkit', 'atf68cawqe', 'm74006xplh', 'did7ayzh7z', 'kh67ey65hl', 'bru91s7d68', 've0ebxq620', 'irhl4zo3lu',
    'yo32drds2u', 'ffv6ygenkg', 'x370zr0v88', 'd31znlyylh', '85y9jx9o86', '2dvxzc8jrd', '7tn35swync', 'piajc9ws3q',
    'eq1z7hu7k3', '1fjsmv778r', '5obxbuubdk', '8wzlxmpbz0', '06sdi34nnm', '65i2zmqjqy', 'ha65d656ik', 'rbq68qstjy',
    '0yj8abtf8t', 'dcn46ntz0p', 'z501bkm67b', 'se47rgbsyd', 'm2w1k92a5a', 'cb9eugktzu', 'mvfn7g0qwu', 'do3ymz05gs',
    'b9u9sk93vy', 'xel4l5p8v6', 'tffxprqn1o', 'x48fzxqgqn', 'bke9bjpf9r', 'l72ae71syd', 'aamr2zeydd', 'z8f2ejslwi',
    '9fv82qyced', 'uc9n9xrv1e', 'alc6p2yimn', '5jcghtltbx', '4lgdy5a4ta', 'ftf9nvfej5', 'eoaer82enz', 'y8lurw2s08',
    'dody7z00ta', '89m3dkhxyy', 'jgofpoxdpd', 'w56izt36gs', 'huoz3b61jz', 'ipk3zzo75v', 'z7utqlcs8m', 's65fsavh28',
    '0s9rpkf53u', 'gy2dprpjxa', 'dnqkkwgd05', '3jzeliplnn', 'q66wswcsev', '3ux82tu34t', '3iy8d4dyxq', 'wil5zsif77',
    'e95rv48otc', 'kggb6goupa', '2p40zfuzsl', '8l2orhbsvl', 's9pcborf7y', 'wq82qwsrm8', 'gs6ng54l0l', 'crck62eltz',
    'e097ctjxvg', 'mdwfmpr3c8', '4jn67lbqhb', '28p358130o', 'hhs8xx201x', 'n7v6g34e0m', 'sz4ikuqw2e', 'zbg97me31s',
    'cod914fnx4', '6x2xfo2ygq', '3c5c90zv36', 'w0qm9q3ey5', 'o75vmzxbse', 'rp1zjtuksu', 'tnn49fvk1p', 'wpbo51inbu',
    'j0cf8hrkjw', 'mnw7zjieww', '7grlc2aopw', 'fr8d5xn4ih', 'bluyc51w9k', 's1pxf3rg0q', 'ubq4rpw1wu', '3lxxctblxe',
    '0kg5dhdk4s', 'd2bjb8l93k', 'ptserhps1f', 'y4ha3liwwc', 'wcl42s7y1r', 'uafxee1kjk', 'ju8og5msxm', 'kbaaq0n44y',
    'ebp3zxjwi7', '6p9l9b6dwr', 'pqtlskkq5f', 'gkr9p9i0d9', 'h3c5drx2ki', 'npykxas4c8', 'b92fu80uys', 'wvxqqsz1w8',
    'df1fjpfewv', '8lxqk97zyh', 'wxl14n9fo3', '2s666tklgo', 'kjv0lvcezv', '9s8i2290qg', 'fzdfz5gkld', '6csrigfo3r',
    '8q6evmyn2i', 'jpxofd3gmu', '7sys4laq6v', 'sxj3j42ygo', 'unan8n2rta', 'ap74o1i50d', 'at0f17pekv', '4cylbd0s3d',
    'b3fudy3nhc', 'kam577w6oq', 'goxg9wwonn', 'trwck67730', '8om7bxkb13', 'bp9xhteuj9', 'sinjn961eq', 'i1xcccg4ok',
    'mf2902f6ro', '7qrer294i2', 'fom5bv844r', 'voijxc93c5', 'zyzqoa0xlj', 'tyazlz2ugv', 'qw5vx1k0rw', 'j540tejlka',
    '1yynvi0pfn', 'v7qghw0r67', 'an1boou6nc', 'bb0wdwlexf', 'oqwz3j1hh1', 'lvakruz2kz', 'oesivb520y', 'em9wf77jqc',
    'b2n32mdq0u', 'xukeok4qlf', '4eiujvozyl', 'tjt5illw2p', 'jrzjjh3g1r', '8gxzzbv7w8', 'p77d58vew4', 'tx3m8otqgw',
    's2e076ohb1', 'g12sx19cjl', 'a1lmo1xxj5', 'gvea18bccm', 'zz00qbgu3m', '5ntior2ch1', 'fd6c02qezd', 'jfc5ld8k0t',
    'pymg8s4d83', 'dp8imn8ze4', '905tkjc9ui', 'ay8mkydpb2', 'h0gzv3styy', '0uhj5zj3ce', 'z6qjin0qqt', 'jjm889wxay',
    'd5c5afq5ue', '1s5t2e9cdw', '1r30xz0nrz', 'ch53nwd306', '2r051ups8k', 'b5yn1tzci0', 'f0csx339lz', '90fb7u7080',
    '9wi6zvt4h3', 'dcc90z03il', 'bh6mqc68g2', '4zv87269nu', 'n0o2wcoq4b', 'eou3xy3ht8', '7bgm052mqy', 'mtje7qh8jp',
    'puunytmd6z', 'd91rup9lkj', 'wi7jemi7yd', 'wdtj8btn9z', 'cc3pkwk0li', 'rr1b9eseyh', 'nr79m4u3lq', 'hruffydq1k',
    'gcbnox9x1n', '56ps8t229t', '64kpdgz5dw', 'uhn8agncvy', '8afjjg2cyc', 'dsv4lhwhn7', 'a5ajz0rgrm', '59ts5ymavk',
    'qlr2bgxwhf', 'v9dax6z7sh', '98p045fecs', 'xymxg866b5', '04ut085ltq', 'u8a9zwvbvy', '9jnlo94889', 'h3zr1w3onf',
    '6buadg1d1d', 'fjuws8vv9r', '31qfj8revn', 'jv49djokvb', 'y3taw4yt70', 't4fsvfyfie', 'jdh2e2tnm5', 'sdt0io3tms',
    'uu47abw0dl', '1md1xx6bfm', 'h1r681fij5', '7lxndqnhv2', '2kcssk11ho', '01lxg391at', 'lsf2z6v5tj', 'd1oyno57ys',
    '0dc9giy2yh', 'z5u5cghmkb', 'vyieqcw386', 'rc4d2ysdn3', 'an8tdlt5m3', 'hkwq608ywv', 'wbs71h9u0v', 'g7bwcljlcr',
    '3eds7bimpz', 'wanieo2pgb', 'vx9o5k2lh9', 'axc0e0c7le', 'oipdyk6cbe', 'kwa7t0ray4', 'y8t8eyooox', 'uwkjlomg44',
    'rbfxjlia3f', 'l2i17obgyi', '3ojubvieb2', 'zqamwvhq33', 'wo1cwyb0pt', 'kqep0e5n7r', 'x91qwoa597', 'lstx3iocs6',
    'b1pjnbqrdr', 'yr4rr5qu3e', 'g1dool8ykk', '9j574ajtba', 'nj42kxv7ks', '1lm8w9z90v', 'e4zla8l7mn', 'kvocbcbsgv',
    'wzqaeg6y2w', 'faanf09q00', 'mfe79ij88l', 'zd6rk7oghq', 'ymikxh8lgo', 'gfha154t8w', 'fzioir534y', 'xqp5uvd92d',
    'lcolftissc', 'jupncpuwyt', 'mrt5myvr4d', 'v4h7pminh1', 'qbd1goytzz', 'ghswt6r5vl', 'm2cnbqdb8a', 'zj1c75vf67',
    'xkmcvrx1mq', 'vf7lmnbqi8', '3wj0t308sq', 'tmhw6nvg5h', '6wbq2lxcre', 'x68c1lnsxf', 'm26yvo17yk', '33cdlwuhk3',
    'rtosgwch6q', '40egrp3age', 'd3afiaj868', '49wngttuec', '0bt4dl1f49', 'wio5uzi5pv', 'htn7736sjm', 'e9tplbqz9l',
    'sff3srqzre', 'jn50sa0z7r', 'vh510bocta', '2d0e97b93g', 'o9mnopmdba', 'tur8btuhld', 'econ137w5q', 'q5gaaxmql4',
    '0tvekrqwv1', 'hoh508yntk', 'f5ajf2opaf', 'khxvl4gc8r', 'e658trpvpa', 'uoh6yyyf8h', 'ulfvn5xmgm', 'f7v2h4cqc1',
    'gbq6n25ye0', 'uoe8sjxm4h', 'vpkyna241n', '4ctd3fq57q', '5maxp2o3xc', '2fl1ip4ifr', '5tqallyce2', '3eglymq80r',
    'n4zypvvycn', 'syc1ih4v1p', 'dnhrcp6pao', 'onvfob000r', 'txwb6uz3dl', '8vs5o45xc7', 'f7v4x0bn28', 'amjxzogjym',
    'b8fbfuqs4s', 'x0djmmux2b', 'd5n724qriv', 'o7sz2gn8ep', 'xq3vzel7mw', '1gypkp7d7r', 'ilkg1bbhki', 'hdyubeknoa',
    'mmknbwc3qf', '64um7iv24n', 'ak5swmi6nj', 'xjszizqtbx', 'rd28uz2wuy', 'uj5cin0oda', '49vx0nuuv7', 'zzca4ye7cv',
    'ao967xr7u7', 'ef1nwshoqj', 'rz66emtdwq', '3hnh7lscve', 'i2gqabmuv4', 'sc05m6ppyd', 'ozlj879y4f', '5kt0ywwp4z',
    'mmto0f0aee', 'kvjaqv4fa1', '7h0zy2vp65', 'wxoblob6nt', 'fp02cubifm', '455frmn9hl', 'x8esaf2cwr', 'p5hwr8pgcv',
    'wwo5dsfe25', '03vbtkww28', 'bt5y2iq3kp'),
    'palette': [(0, 0, 0), (116, 134, 32), (107, 208, 177), (18, 238, 229), (250, 91, 150), (72, 215, 83), (5, 1, 233),
                (187, 118, 118), (158, 1, 133), (54, 231, 96), (134, 254, 136), (193, 219, 87), (191, 8, 163),
                (209, 17, 101), (101, 217, 138), (45, 42, 11), (211, 185, 197), (152, 20, 75), (209, 158, 99),
                (11, 148, 207), (251, 120, 251), (232, 101, 196), (231, 7, 62), (163, 15, 12), (230, 225, 74),
                (90, 111, 21), (162, 202, 247), (156, 140, 64), (59, 124, 46), (137, 128, 159), (182, 224, 170),
                (215, 42, 198), (197, 95, 39), (20, 247, 81), (72, 142, 175), (13, 229, 195), (150, 74, 177),
                (115, 167, 242), (31, 20, 164), (27, 211, 106), (226, 28, 137), (161, 182, 152), (173, 178, 29),
                (217, 103, 220), (147, 77, 23), (121, 10, 223), (126, 184, 41), (200, 118, 247), (173, 85, 238),
                (234, 25, 134), (209, 190, 8), (114, 4, 193), (52, 158, 99), (170, 221, 23), (121, 54, 31),
                (84, 149, 148), (181, 58, 226), (200, 16, 100), (235, 66, 14), (31, 247, 51), (231, 55, 128),
                (105, 182, 47), (76, 70, 195), (43, 77, 130), (176, 51, 31), (127, 201, 98), (224, 53, 255),
                (222, 36, 204), (124, 69, 189), (228, 245, 200), (149, 67, 244), (197, 39, 211), (224, 116, 220),
                (83, 235, 100), (177, 186, 41), (3, 204, 212), (223, 215, 119), (185, 12, 164), (232, 246, 115),
                (101, 229, 164), (238, 231, 93), (195, 85, 242), (162, 18, 116), (93, 185, 225), (246, 191, 183),
                (139, 84, 168), (223, 155, 144), (209, 82, 95), (94, 167, 144), (126, 64, 238), (52, 20, 180),
                (44, 224, 169), (55, 61, 157), (83, 65, 8), (218, 76, 249), (181, 24, 49), (242, 139, 171),
                (30, 222, 221), (212, 141, 126), (118, 42, 68), (30, 44, 35), (42, 27, 208), (17, 47, 162),
                (33, 32, 178), (78, 250, 238), (187, 22, 10), (172, 101, 221), (180, 103, 100), (202, 33, 166),
                (9, 158, 125), (44, 122, 1), (204, 227, 244), (2, 240, 144), (66, 92, 194), (144, 197, 27),
                (148, 144, 171), (175, 7, 118), (101, 180, 59), (187, 200, 51), (168, 203, 117), (62, 143, 141),
                (215, 137, 32), (85, 185, 130), (228, 233, 154), (28, 196, 224), (2, 100, 112), (177, 42, 75),
                (60, 184, 167), (109, 174, 217), (251, 131, 181), (176, 107, 253), (37, 196, 77), (81, 65, 66),
                (235, 21, 123), (40, 177, 77), (7, 77, 221), (96, 35, 242), (196, 5, 7), (103, 241, 170), (53, 67, 129),
                (82, 240, 117), (34, 249, 150), (156, 238, 248), (190, 53, 146), (96, 188, 144), (170, 123, 113),
                (201, 196, 88), (170, 24, 134), (8, 78, 161), (12, 141, 11), (184, 208, 221), (161, 218, 67),
                (82, 16, 187), (226, 210, 248), (255, 157, 12), (46, 211, 178), (227, 33, 106), (117, 51, 0),
                (95, 65, 197), (152, 182, 197), (17, 176, 55), (196, 109, 30), (79, 134, 22), (28, 14, 92),
                (250, 99, 198), (151, 29, 114), (86, 128, 20), (120, 25, 171), (90, 222, 161), (211, 6, 19),
                (68, 149, 81), (238, 133, 77), (242, 180, 188), (27, 16, 70), (95, 214, 168), (206, 98, 3),
                (119, 29, 22), (52, 12, 69), (37, 45, 234), (253, 87, 70), (13, 116, 61), (198, 227, 217),
                (150, 235, 83), (171, 144, 92), (204, 75, 19), (147, 103, 228), (17, 114, 42), (156, 78, 139),
                (146, 113, 205), (214, 161, 96), (82, 27, 206), (121, 202, 44), (189, 103, 2), (187, 227, 164),
                (110, 147, 104), (237, 149, 66), (234, 41, 90), (37, 133, 39), (144, 169, 45), (15, 154, 232),
                (150, 192, 104), (30, 151, 5), (49, 41, 23), (35, 89, 196), (166, 109, 142), (109, 6, 187),
                (148, 245, 156), (173, 203, 28), (149, 68, 21), (69, 144, 117), (14, 182, 142), (102, 131, 129),
                (222, 244, 75), (128, 107, 1), (51, 23, 210), (38, 249, 213), (194, 93, 245), (58, 137, 20),
                (205, 83, 93), (205, 199, 42), (123, 40, 191), (67, 95, 47), (245, 228, 56), (194, 42, 128),
                (82, 249, 121), (193, 245, 0), (35, 72, 139), (189, 82, 73), (148, 49, 35), (44, 13, 230),
                (109, 184, 125), (175, 138, 216), (176, 0, 117), (10, 189, 164), (152, 127, 246), (44, 37, 137),
                (125, 178, 23), (227, 8, 173), (7, 46, 106), (196, 212, 131), (143, 146, 54), (26, 249, 112),
                (23, 138, 80), (11, 61, 156), (26, 182, 177), (6, 36, 113), (69, 162, 113), (14, 221, 113),
                (232, 211, 236), (228, 62, 54), (227, 3, 30), (228, 43, 24), (185, 201, 81), (134, 142, 37),
                (93, 49, 182), (75, 217, 60), (61, 212, 75), (102, 14, 4), (96, 117, 249), (103, 239, 125),
                (145, 201, 70), (140, 28, 85), (20, 23, 129), (46, 181, 233), (89, 4, 152), (162, 92, 51), (214, 8, 60),
                (211, 135, 88), (69, 30, 207), (25, 76, 70), (119, 240, 126), (31, 179, 23), (148, 152, 54),
                (88, 93, 171), (70, 217, 246), (154, 210, 35), (137, 163, 67), (172, 138, 235), (116, 10, 225),
                (213, 117, 230), (3, 49, 127), (99, 0, 146), (101, 0, 169), (81, 6, 239), (172, 156, 228),
                (92, 88, 165), (61, 13, 125), (251, 85, 190), (227, 87, 197), (166, 51, 115), (232, 209, 181),
                (111, 142, 217), (118, 113, 227), (72, 44, 34), (164, 51, 126), (75, 178, 221), (113, 114, 6),
                (186, 197, 244), (171, 62, 69), (161, 69, 133), (55, 166, 237), (174, 41, 192), (72, 1, 107),
                (158, 148, 10), (176, 207, 10), (41, 235, 254), (97, 198, 55), (149, 252, 108), (112, 90, 113),
                (205, 33, 198), (76, 238, 72), (105, 48, 39), (234, 184, 179), (159, 152, 163), (219, 154, 15),
                (160, 154, 136), (90, 22, 236), (187, 245, 92), (122, 99, 195), (176, 229, 18), (238, 13, 122),
                (120, 244, 81), (16, 87, 63), (68, 98, 237), (45, 244, 249), (170, 184, 160), (159, 135, 248),
                (213, 127, 191), (20, 153, 171), (138, 75, 172), (232, 87, 125), (234, 238, 90), (68, 203, 98),
                (250, 49, 205), (32, 78, 11), (128, 88, 87), (78, 20, 185), (144, 94, 94), (115, 131, 200),
                (203, 247, 141), (248, 211, 212), (78, 201, 193), (87, 98, 34), (187, 195, 32), (38, 21, 188),
                (202, 136, 67), (205, 208, 150), (28, 226, 39), (119, 135, 226), (188, 115, 203), (61, 213, 227),
                (3, 242, 59), (199, 48, 195), (204, 155, 121), (44, 218, 0), (181, 14, 174), (220, 126, 104),
                (205, 196, 24), (40, 44, 250), (26, 64, 85), (113, 140, 170), (92, 102, 161), (227, 42, 203),
                (12, 92, 74), (206, 221, 42), (155, 236, 151), (9, 144, 23), (96, 153, 176), (5, 234, 188),
                (100, 248, 50), (184, 213, 88), (119, 61, 190), (123, 233, 135), (102, 70, 37), (121, 102, 137),
                (11, 168, 167), (214, 104, 179), (41, 77, 185), (160, 14, 201), (148, 173, 215), (146, 21, 199),
                (92, 175, 15), (73, 165, 78), (223, 88, 75), (85, 56, 26), (51, 42, 110), (59, 9, 215), (76, 75, 217),
                (145, 127, 211), (236, 110, 48), (79, 27, 254), (103, 201, 161), (128, 149, 196), (89, 216, 244),
                (238, 122, 179), (98, 66, 123), (167, 77, 139), (61, 171, 9), (50, 33, 213), (35, 3, 180),
                (129, 125, 87), (164, 55, 189), (252, 63, 64), (250, 93, 112), (152, 203, 72), (83, 151, 154),
                (47, 48, 43), (162, 47, 154), (21, 58, 214), (10, 131, 240), (87, 173, 71), (94, 137, 223),
                (116, 215, 72), (112, 14, 126), (176, 101, 198), (85, 98, 11), (134, 204, 65), (232, 198, 129),
                (198, 88, 51), (49, 246, 3), (40, 78, 87), (12, 221, 227), (222, 71, 102), (27, 140, 177),
                (139, 253, 112), (230, 16, 212), (235, 44, 33), (74, 133, 230), (54, 193, 15), (173, 60, 189),
                (100, 112, 148), (94, 2, 74), (183, 238, 238), (120, 48, 185), (78, 82, 117), (80, 34, 47),
                (124, 50, 51), (146, 148, 203), (218, 138, 55), (188, 248, 86), (61, 153, 12), (130, 250, 27),
                (11, 222, 221), (252, 32, 23), (59, 201, 172), (53, 240, 51), (133, 144, 94), (79, 138, 97),
                (140, 49, 253), (142, 202, 240), (57, 217, 170), (204, 216, 103), (100, 43, 32), (195, 108, 94),
                (188, 249, 49), (40, 82, 69), (255, 250, 130), (77, 30, 166), (161, 114, 28), (108, 78, 136),
                (239, 245, 179), (237, 42, 5), (111, 53, 133), (173, 74, 98), (239, 148, 177), (226, 52, 15),
                (112, 198, 126), (221, 133, 255), (9, 58, 210), (99, 152, 106), (140, 154, 16), (254, 85, 150),
                (27, 136, 16), (200, 197, 177), (156, 121, 110), (62, 207, 24), (178, 227, 66), (12, 89, 229),
                (22, 131, 19), (158, 75, 149), (16, 212, 5), (14, 120, 132), (36, 114, 183), (189, 113, 180),
                (218, 23, 240), (254, 19, 11), (11, 178, 48), (93, 94, 210), (175, 187, 101), (157, 237, 35),
                (193, 240, 209), (74, 167, 233), (43, 24, 140), (52, 70, 213), (35, 87, 196), (62, 175, 78),
                (243, 4, 229), (210, 103, 93), (147, 206, 238), (79, 202, 209), (88, 110, 221), (7, 102, 94),
                (192, 54, 211), (78, 199, 60), (124, 145, 87), (235, 242, 1), (191, 81, 38), (42, 32, 183),
                (151, 104, 55), (123, 71, 193), (31, 24, 129), (119, 147, 170), (77, 166, 219), (80, 190, 72),
                (161, 72, 210), (41, 11, 54), (10, 39, 200), (245, 107, 25), (17, 248, 88), (169, 159, 36),
                (226, 238, 18), (10, 254, 110), (175, 181, 194), (230, 113, 11), (176, 189, 196), (201, 126, 1),
                (39, 28, 138), (178, 241, 95), (190, 117, 168), (107, 81, 108), (197, 119, 99), (49, 95, 38),
                (12, 186, 135), (18, 66, 11), (85, 211, 13), (232, 107, 204), (75, 88, 48), (104, 144, 3),
                (6, 239, 185), (42, 150, 32), (71, 22, 204), (126, 25, 131), (69, 24, 246), (140, 65, 121),
                (193, 203, 188), (45, 64, 6), (196, 109, 61), (198, 223, 42), (0, 50, 134), (201, 129, 68),
                (45, 74, 224), (199, 164, 173), (247, 223, 204), (15, 237, 200), (173, 144, 61), (8, 118, 14),
                (109, 219, 192), (192, 142, 137), (50, 92, 74), (75, 213, 137), (106, 211, 77), (126, 222, 192),
                (174, 147, 237), (104, 234, 239), (224, 168, 91), (59, 15, 173), (123, 166, 195), (20, 232, 37),
                (138, 21, 175), (140, 116, 21), (112, 254, 191), (222, 96, 57), (87, 242, 182), (15, 47, 244),
                (28, 99, 79), (183, 107, 44), (101, 242, 154), (148, 149, 128), (136, 4, 244), (225, 254, 8),
                (106, 47, 214), (235, 110, 211), (10, 84, 120), (102, 59, 87), (158, 79, 16), (193, 183, 36),
                (78, 227, 8), (121, 211, 119), (158, 245, 150), (136, 90, 137), (95, 201, 143), (7, 182, 162),
                (44, 44, 125), (219, 85, 94), (164, 172, 171), (211, 143, 186), (89, 209, 124), (220, 127, 157),
                (154, 174, 252), (196, 235, 68), (190, 169, 0), (138, 185, 205), (65, 23, 182), (12, 208, 169),
                (22, 96, 122), (45, 155, 202), (95, 146, 113), (236, 143, 54), (73, 159, 11), (164, 171, 41),
                (4, 131, 228), (215, 9, 129), (37, 66, 211), (123, 1, 225), (194, 13, 179), (125, 91, 169),
                (77, 13, 51), (221, 222, 243), (81, 123, 144), (114, 202, 166), (45, 136, 209), (207, 243, 185),
                (160, 198, 36), (163, 240, 217), (119, 219, 176), (196, 148, 209), (39, 253, 169), (90, 142, 1),
                (12, 23, 121), (36, 203, 222), (126, 110, 96), (10, 105, 15), (31, 104, 16), (44, 121, 39),
                (209, 156, 15), (204, 146, 27), (90, 125, 158), (251, 235, 155), (137, 205, 47), (245, 131, 9),
                (100, 182, 119), (203, 163, 105), (213, 190, 167), (87, 226, 136), (244, 4, 37), (26, 70, 223),
                (187, 155, 129), (20, 12, 15), (50, 32, 150), (102, 152, 185), (54, 59, 78), (79, 65, 188),
                (123, 202, 96), (55, 42, 143), (232, 164, 108), (149, 204, 47), (144, 167, 242), (33, 190, 63),
                (67, 149, 134), (66, 240, 179), (98, 182, 238), (21, 74, 83), (183, 41, 47), (77, 62, 127),
                (214, 35, 156), (188, 176, 180), (216, 33, 156), (97, 251, 241), (23, 213, 171), (208, 255, 66),
                (190, 155, 52), (28, 115, 154), (72, 200, 102), (201, 3, 157), (213, 72, 82), (178, 210, 115),
                (185, 139, 43), (75, 15, 114), (240, 79, 39), (75, 84, 85), (97, 210, 129), (62, 145, 41),
                (117, 40, 166), (24, 123, 254), (103, 108, 195), (77, 93, 78), (19, 209, 179), (159, 54, 26),
                (123, 36, 98), (70, 23, 36), (48, 51, 215), (88, 156, 243), (234, 164, 149), (195, 30, 166),
                (229, 174, 42), (152, 161, 192), (50, 201, 99), (198, 111, 58), (249, 71, 236), (245, 223, 139),
                (79, 61, 59), (245, 199, 179), (239, 184, 205), (91, 243, 92), (41, 165, 138), (134, 130, 78),
                (80, 121, 19), (191, 51, 66), (19, 69, 72), (8, 190, 106), (29, 180, 110), (92, 0, 230), (159, 178, 87),
                (159, 132, 73), (110, 182, 49), (159, 128, 250), (23, 5, 41), (126, 69, 253), (200, 42, 223),
                (192, 193, 182), (150, 86, 174), (82, 127, 56), (5, 212, 126), (67, 51, 31), (141, 123, 103),
                (125, 247, 31), (167, 88, 86), (64, 134, 192), (108, 126, 239), (93, 96, 71), (147, 40, 217),
                (8, 196, 145), (252, 120, 183), (81, 189, 173), (76, 34, 144), (41, 193, 24), (89, 134, 173),
                (203, 130, 155), (45, 41, 170), (237, 7, 178), (124, 97, 110), (50, 156, 123), (233, 244, 4),
                (55, 223, 25), (7, 33, 199), (158, 113, 208), (129, 153, 39), (211, 166, 182), (90, 67, 4),
                (92, 117, 70), (98, 10, 162), (155, 78, 26), (184, 41, 95), (87, 101, 104), (207, 54, 21),
                (105, 84, 35), (121, 181, 165), (54, 242, 126), (73, 87, 85), (80, 41, 1), (54, 255, 58),
                (241, 232, 227), (15, 43, 232), (107, 48, 11), (147, 150, 189), (11, 164, 101), (229, 79, 84),
                (181, 68, 16), (61, 17, 6), (171, 53, 71), (38, 3, 198), (102, 210, 206), (73, 218, 233),
                (12, 219, 214), (21, 216, 180), (216, 125, 103), (25, 210, 240), (48, 76, 164), (90, 8, 237),
                (30, 149, 212), (44, 3, 6), (60, 170, 192), (6, 48, 101), (166, 235, 78), (114, 214, 152),
                (253, 129, 74), (249, 26, 227), (208, 121, 208), (149, 166, 47), (253, 92, 115), (205, 19, 66),
                (10, 79, 62), (45, 5, 74), (50, 98, 88), (217, 22, 174), (116, 67, 234), (138, 161, 220),
                (192, 232, 116), (78, 77, 88), (57, 235, 10), (35, 5, 52), (219, 200, 72), (168, 101, 5),
                (160, 142, 88), (207, 244, 215), (151, 145, 130), (2, 72, 31), (205, 181, 230), (102, 123, 196),
                (146, 215, 250), (180, 180, 204), (91, 178, 161), (85, 29, 69), (134, 117, 168), (118, 96, 15),
                (202, 210, 166), (216, 182, 30), (23, 191, 84), (174, 3, 4), (170, 59, 113), (192, 75, 174),
                (154, 173, 166), (225, 24, 37), (167, 240, 94), (5, 231, 45), (11, 46, 74), (12, 95, 80),
                (175, 201, 25), (252, 17, 52), (246, 135, 236), (79, 53, 2), (197, 87, 137), (121, 255, 14),
                (246, 167, 95), (151, 156, 88), (38, 145, 51), (119, 129, 121), (112, 100, 145), (245, 241, 244),
                (21, 172, 4), (198, 138, 183), (162, 39, 72), (19, 8, 242), (28, 88, 97), (175, 71, 118),
                (65, 118, 254), (7, 225, 71), (98, 156, 106), (208, 218, 41), (182, 17, 135), (21, 147, 27),
                (180, 49, 113), (198, 127, 114), (148, 97, 219), (211, 2, 154), (124, 65, 131), (250, 218, 116),
                (189, 95, 60), (178, 187, 227), (252, 143, 96), (183, 152, 97), (178, 61, 246), (166, 70, 63),
                (21, 180, 170), (169, 126, 10), (176, 89, 89), (93, 83, 96), (189, 134, 102), (17, 195, 97),
                (238, 90, 3), (143, 153, 41), (195, 55, 169), (199, 254, 44), (95, 101, 205), (21, 156, 204),
                (34, 244, 86), (8, 5, 67), (154, 48, 121), (217, 88, 17), (125, 142, 134), (232, 106, 219),
                (73, 245, 144), (55, 150, 78), (161, 206, 119), (127, 161, 241), (236, 29, 228), (77, 134, 184),
                (97, 103, 62), (171, 229, 91), (239, 79, 148), (66, 60, 121), (92, 61, 21), (81, 132, 180),
                (144, 161, 11), (197, 145, 23), (189, 100, 159), (154, 88, 145), (131, 81, 225), (233, 207, 38),
                (185, 45, 163), (213, 197, 23), (141, 199, 188), (185, 51, 208), (48, 68, 246), (137, 150, 54),
                (133, 116, 242), (12, 62, 49), (146, 68, 10), (131, 111, 68), (120, 75, 250), (30, 86, 152),
                (194, 91, 101), (138, 60, 109), (82, 171, 2), (221, 124, 153), (7, 241, 17), (162, 30, 141),
                (202, 250, 65), (218, 23, 92), (166, 31, 183), (18, 174, 224), (52, 221, 207), (173, 248, 3),
                (34, 54, 53), (129, 217, 93), (1, 98, 6), (15, 214, 63), (64, 198, 26), (162, 60, 120), (227, 110, 12),
                (116, 89, 141), (168, 67, 211), (187, 96, 210), (71, 58, 226), (225, 115, 5), (176, 82, 223),
                (145, 225, 152), (204, 46, 102), (184, 101, 18), (109, 121, 116), (122, 144, 114), (101, 146, 181),
                (32, 197, 81), (211, 96, 50), (12, 60, 51), (62, 205, 0), (130, 254, 200), (12, 41, 229), (53, 7, 82),
                (34, 171, 193), (77, 63, 41), (126, 180, 1), (233, 151, 94), (235, 205, 252), (246, 107, 222),
                (151, 211, 84), (54, 88, 4), (178, 26, 205), (144, 102, 17), (52, 149, 91), (208, 150, 212),
                (86, 25, 192), (121, 26, 215), (89, 250, 166), (217, 141, 60), (239, 2, 221), (238, 200, 88),
                (118, 13, 38), (37, 172, 35), (227, 246, 16), (161, 65, 24), (207, 25, 175), (106, 147, 79),
                (106, 220, 30), (215, 253, 248), (128, 196, 20), (202, 32, 242), (133, 179, 32), (110, 65, 24),
                (148, 144, 153), (153, 177, 196), (30, 182, 152), (13, 161, 119), (242, 154, 10), (0, 45, 185),
                (82, 210, 72), (25, 96, 70), (182, 180, 171), (230, 206, 178), (186, 116, 91), (214, 42, 223),
                (19, 57, 64), (215, 154, 217), (24, 147, 165), (232, 64, 251), (111, 37, 190), (162, 234, 138),
                (86, 240, 28), (149, 165, 244), (239, 21, 54), (222, 193, 131), (123, 80, 138), (27, 128, 97),
                (17, 101, 234), (44, 40, 183), (1, 27, 120), (112, 231, 50), (166, 38, 195), (128, 215, 24),
                (21, 251, 99), (96, 204, 221), (5, 172, 104), (76, 59, 222), (52, 250, 78), (207, 113, 74),
                (27, 38, 216), (122, 33, 58), (104, 73, 51), (60, 216, 7), (211, 44, 115), (168, 214, 232),
                (200, 197, 57), (80, 253, 143), (126, 113, 81), (111, 83, 211), (233, 8, 100), (32, 211, 76),
                (212, 118, 4), (99, 42, 150), (18, 255, 43), (178, 231, 19), (35, 115, 169), (173, 241, 7),
                (193, 180, 63), (30, 247, 227), (6, 94, 20), (221, 179, 250), (181, 91, 176), (158, 145, 95),
                (94, 70, 29), (39, 119, 202), (151, 2, 120), (35, 9, 113), (160, 229, 32), (90, 213, 83),
                (116, 182, 83), (122, 97, 140), (108, 116, 140), (17, 125, 151), (211, 5, 8), (253, 185, 78),
                (2, 166, 164), (157, 3, 82), (17, 10, 197), (108, 39, 43), (205, 12, 97), (185, 13, 155),
                (154, 211, 67), (149, 175, 240), (46, 157, 18), (223, 121, 196), (202, 229, 209), (65, 162, 203),
                (201, 151, 57), (106, 115, 196), (78, 52, 71), (161, 126, 112), (100, 174, 208), (158, 236, 35),
                (228, 21, 234), (245, 65, 202), (178, 74, 82), (98, 245, 157), (134, 185, 124), (21, 199, 36),
                (239, 217, 155), (204, 209, 200), (13, 190, 78), (24, 26, 75), (118, 86, 87), (164, 141, 177),
                (51, 50, 19), (221, 202, 105), (60, 141, 114), (116, 252, 32), (227, 65, 232), (194, 89, 111),
                (105, 255, 238), (22, 9, 229), (23, 83, 138), (14, 226, 105), (20, 92, 132), (106, 1, 155),
                (249, 240, 87), (37, 97, 233), (134, 81, 153), (169, 213, 214), (248, 34, 186), (82, 60, 67),
                (144, 146, 5), (57, 237, 8), (246, 36, 190), (150, 69, 49), (14, 148, 80), (1, 246, 231),
                (147, 153, 147), (44, 250, 55), (25, 177, 130), (36, 149, 227), (91, 46, 193), (239, 32, 166),
                (207, 210, 55), (52, 66, 244), (85, 82, 186), (30, 223, 195), (105, 145, 192), (58, 176, 218),
                (56, 228, 222), (195, 76, 193), (192, 62, 101), (213, 152, 221), (58, 153, 201), (95, 99, 113),
                (216, 215, 164), (189, 217, 215), (7, 120, 217), (148, 158, 249), (6, 84, 66), (82, 253, 166),
                (97, 76, 46), (173, 196, 112), (111, 178, 16), (15, 183, 42), (78, 164, 68), (54, 166, 254),
                (205, 40, 209), (159, 63, 22), (123, 9, 3), (241, 180, 145), (82, 146, 219), (163, 77, 136),
                (80, 44, 60), (173, 24, 231), (80, 86, 45), (214, 138, 41), (200, 16, 28), (40, 166, 67),
                (120, 191, 118), (244, 205, 214), (103, 252, 96), (223, 5, 61), (244, 127, 203), (193, 15, 131),
                (158, 100, 15), (194, 65, 105), (176, 195, 46), (187, 87, 82), (227, 185, 67), (113, 139, 118),
                (249, 97, 0), (154, 188, 154), (125, 188, 175), (239, 90, 226), (106, 141, 134), (15, 142, 20),
                (182, 57, 156), (53, 62, 184), (192, 65, 147), (11, 124, 224), (36, 148, 253), (180, 204, 52),
                (41, 166, 28), (3, 202, 154), (205, 251, 96), (71, 34, 142), (206, 160, 110), (207, 23, 26),
                (96, 46, 21), (58, 234, 2), (221, 153, 129), (235, 229, 5), (173, 232, 180), (148, 41, 207),
                (189, 168, 179), (242, 220, 88), (57, 151, 15), (165, 50, 102), (203, 38, 112), (163, 73, 196),
                (130, 160, 94), (165, 17, 232), (62, 199, 214), (98, 251, 92), (217, 120, 243), (164, 24, 143),
                (220, 95, 165), (211, 53, 90), (33, 115, 30), (162, 102, 248), (140, 204, 42), (255, 91, 186),
                (1, 208, 8), (149, 102, 193), (169, 197, 226), (4, 210, 49), (73, 106, 81), (229, 140, 121),
                (84, 175, 100), (163, 19, 136), (174, 12, 22), (63, 225, 72), (64, 170, 105), (243, 39, 28),
                (110, 171, 153), (170, 60, 118), (152, 98, 78), (158, 72, 231), (178, 118, 89), (106, 108, 216),
                (210, 219, 15), (43, 161, 118), (67, 245, 233), (125, 24, 96), (18, 30, 120), (182, 97, 141),
                (199, 83, 118), (23, 164, 242), (182, 184, 82), (226, 198, 162), (186, 70, 130), (2, 10, 242),
                (235, 9, 249), (233, 241, 193), (160, 153, 78), (88, 190, 186), (30, 80, 48), (107, 185, 127),
                (68, 29, 11), (137, 224, 183), (47, 136, 60), (5, 88, 139), (198, 195, 131), (228, 190, 4),
                (14, 216, 254), (137, 189, 50), (250, 53, 221), (59, 182, 178), (102, 79, 65), (75, 117, 92),
                (53, 247, 50), (97, 32, 147), (186, 108, 75), (13, 240, 29), (238, 2, 126), (17, 29, 190),
                (190, 162, 249), (9, 36, 26), (56, 145, 165), (98, 218, 149), (138, 32, 195), (225, 114, 168),
                (128, 109, 40), (89, 212, 39), (254, 164, 167), (226, 146, 39), (237, 209, 249), (17, 38, 251),
                (225, 119, 212), (179, 87, 46), (248, 128, 205), (95, 113, 200), (2, 65, 254), (64, 252, 133),
                (66, 126, 35), (94, 162, 194), (41, 207, 216), (157, 205, 97), (191, 193, 236), (233, 147, 16),
                (132, 87, 28), (141, 127, 224), (130, 51, 77), (171, 200, 107), (95, 187, 9), (112, 45, 245),
                (110, 78, 86), (56, 9, 176), (127, 80, 199), (107, 158, 207), (152, 236, 218), (25, 103, 224),
                (15, 68, 21), (40, 146, 218), (80, 54, 131), (201, 114, 48), (82, 247, 62), (129, 224, 223),
                (190, 42, 143), (137, 63, 43), (203, 14, 148), (42, 164, 182), (32, 117, 97), (191, 3, 243),
                (73, 208, 98), (162, 53, 194), (22, 55, 50), (233, 131, 10), (207, 195, 255), (137, 51, 164),
                (178, 235, 236), (216, 170, 146), (48, 34, 38), (162, 12, 76), (140, 202, 137), (251, 218, 151),
                (189, 1, 7), (117, 205, 182), (186, 11, 10), (52, 57, 127), (31, 80, 190), (66, 236, 132),
                (205, 234, 237), (91, 126, 41), (130, 242, 23), (13, 111, 232), (146, 111, 145), (223, 213, 8),
                (214, 154, 162), (242, 160, 80), (142, 15, 86), (240, 194, 6), (184, 95, 28), (176, 69, 33),
                (72, 116, 152), (95, 70, 65), (134, 105, 94), (70, 208, 224), (187, 69, 55), (94, 152, 226),
                (8, 147, 69), (197, 87, 72), (114, 107, 153), (135, 36, 11), (90, 55, 158), (249, 106, 41),
                (34, 56, 233), (192, 205, 9), (111, 94, 173), (104, 12, 99), (42, 67, 140), (180, 185, 244),
                (25, 147, 98), (181, 214, 119), (156, 142, 111), (129, 194, 64), (32, 125, 169), (198, 185, 64),
                (209, 27, 214), (88, 153, 3), (36, 102, 144), (87, 14, 80), (136, 82, 5), (169, 252, 85),
                (143, 205, 79), (100, 80, 62), (174, 148, 144), (233, 141, 142), (197, 18, 21), (42, 229, 64),
                (29, 250, 48), (254, 121, 95), (141, 2, 212), (104, 46, 215), (180, 220, 206), (170, 128, 171),
                (164, 166, 106), (241, 171, 104), (105, 144, 251), (30, 3, 232), (117, 192, 143), (124, 162, 200),
                (253, 137, 23), (173, 106, 184), (81, 231, 223), (220, 136, 249), (228, 161, 15), (64, 171, 207),
                (133, 80, 244), (18, 52, 160), (167, 175, 135), (142, 112, 185), (76, 202, 171), (79, 37, 166),
                (123, 74, 9), (141, 80, 26), (5, 203, 217), (109, 117, 76), (53, 17, 179), (223, 161, 96),
                (124, 169, 225), (52, 172, 202), (162, 67, 90), (175, 134, 177), (124, 255, 155), (94, 50, 208),
                (43, 231, 1), (164, 189, 89), (255, 100, 152), (140, 24, 215), (79, 128, 230), (214, 236, 168),
                (237, 209, 137), (191, 244, 203), (156, 154, 151), (43, 180, 150), (20, 32, 93), (71, 181, 45),
                (80, 50, 109), (118, 210, 176), (153, 208, 112), (77, 180, 2), (137, 120, 211), (224, 160, 128),
                (83, 136, 123), (210, 243, 254), (183, 186, 153), (208, 112, 253), (206, 68, 234), (172, 171, 142),
                (3, 198, 6), (135, 159, 214), (178, 100, 33), (120, 17, 248), (120, 13, 5), (225, 216, 72),
                (45, 91, 250), (192, 179, 174), (186, 76, 16), (183, 106, 110), (169, 89, 55), (164, 3, 99),
                (34, 69, 118), (205, 23, 113), (32, 36, 171), (231, 103, 192), (5, 22, 192), (55, 110, 241),
                (51, 151, 222), (255, 33, 223), (126, 215, 140), (2, 200, 231), (220, 192, 220), (157, 137, 47),
                (135, 197, 138), (113, 106, 203), (3, 240, 211), (124, 155, 109), (142, 64, 39), (205, 246, 153),
                (80, 142, 148), (38, 221, 146), (9, 71, 154), (190, 143, 204), (52, 114, 198), (151, 90, 254),
                (96, 254, 16), (121, 43, 200), (185, 178, 142), (147, 35, 198), (174, 143, 77), (13, 145, 248),
                (53, 63, 243), (90, 121, 94), (14, 111, 179), (74, 43, 14), (14, 150, 93), (236, 233, 56),
                (190, 224, 148), (13, 171, 228), (88, 82, 151), (225, 234, 94), (244, 199, 225), (61, 164, 225),
                (68, 68, 245), (112, 244, 32), (31, 6, 152), (70, 5, 204), (226, 10, 104), (185, 122, 83),
                (184, 15, 222), (173, 62, 152), (64, 205, 211), (141, 215, 187), (127, 85, 51), (16, 216, 34),
                (60, 132, 153), (210, 236, 183), (187, 254, 247), (159, 63, 65), (62, 177, 104), (245, 242, 42),
                (197, 218, 108), (213, 42, 187), (227, 157, 41), (184, 56, 164), (147, 188, 40), (29, 177, 191),
                (73, 24, 147), (194, 72, 134), (168, 201, 231), (129, 95, 169), (104, 76, 104), (187, 10, 206),
                (227, 160, 187), (14, 209, 23), (225, 197, 192), (198, 233, 131), (85, 143, 221), (207, 233, 139),
                (140, 50, 54), (186, 97, 201), (219, 198, 250), (126, 233, 64), (3, 174, 158), (126, 104, 90),
                (246, 141, 64), (31, 180, 125), (13, 178, 33), (179, 107, 181), (36, 136, 219), (109, 221, 92),
                (6, 232, 244), (91, 188, 159), (32, 154, 191), (121, 21, 116), (75, 49, 64), (118, 228, 135),
                (104, 226, 2), (80, 150, 70), (67, 123, 235), (67, 8, 165), (82, 179, 95), (17, 166, 34),
                (127, 225, 173), (118, 210, 5), (36, 142, 160), (160, 19, 220), (237, 154, 216), (111, 211, 196),
                (84, 126, 195), (230, 79, 107), (147, 255, 184), (130, 22, 196), (71, 183, 52), (121, 141, 241),
                (58, 128, 74), (15, 132, 43), (138, 3, 154), (26, 232, 226), (27, 94, 16), (229, 101, 250),
                (207, 111, 152), (173, 110, 229), (77, 232, 240), (127, 52, 136), (121, 88, 207), (93, 68, 182),
                (111, 109, 124), (222, 172, 142), (142, 161, 127), (143, 227, 125), (46, 84, 236), (43, 179, 29),
                (0, 200, 77), (138, 29, 151), (245, 224, 59), (238, 230, 118), (125, 76, 177), (124, 94, 117),
                (169, 127, 118), (69, 100, 226), (74, 164, 50), (125, 64, 197), (90, 46, 195), (229, 47, 26),
                (42, 244, 45), (139, 26, 85), (21, 199, 118), (105, 181, 219), (248, 8, 194), (51, 12, 42),
                (123, 125, 38), (96, 172, 175), (162, 121, 164), (60, 82, 104), (127, 38, 248), (38, 51, 25),
                (54, 251, 185), (14, 5, 50), (145, 175, 209), (118, 204, 237), (172, 44, 204), (251, 209, 160),
                (223, 125, 238), (54, 247, 19), (44, 211, 162), (79, 85, 125), (12, 25, 70), (199, 59, 97),
                (216, 206, 83), (23, 189, 233), (79, 96, 3), (100, 118, 101), (158, 82, 82), (73, 96, 19), (5, 59, 236),
                (228, 148, 89), (27, 239, 160), (166, 101, 148), (165, 152, 85), (79, 62, 178), (94, 80, 222),
                (176, 27, 17), (122, 69, 255), (146, 121, 136), (203, 151, 105), (150, 22, 60), (225, 197, 11),
                (217, 28, 57), (158, 98, 84), (9, 116, 191), (246, 102, 37), (172, 50, 150), (239, 227, 4),
                (185, 10, 170), (79, 120, 14), (67, 143, 78), (122, 26, 240), (178, 184, 86), (108, 20, 246),
                (131, 204, 211), (64, 139, 52), (117, 43, 87), (80, 1, 24), (163, 246, 16), (2, 232, 247),
                (150, 237, 159), (103, 44, 106), (41, 113, 90), (124, 57, 51), (79, 35, 46), (241, 27, 148),
                (59, 63, 162), (249, 220, 133), (58, 160, 43), (192, 187, 189), (113, 158, 36), (85, 36, 36),
                (64, 155, 254), (111, 164, 44), (93, 178, 26), (177, 153, 118), (23, 59, 88), (45, 13, 34),
                (227, 80, 110), (22, 144, 75), (110, 45, 198), (3, 228, 127), (108, 146, 16), (100, 37, 231),
                (50, 133, 147), (185, 208, 109), (30, 129, 236), (197, 83, 129), (139, 105, 117), (229, 160, 144),
                (190, 13, 204), (169, 119, 73), (26, 247, 6), (59, 234, 250), (115, 105, 31), (201, 146, 177),
                (17, 84, 17), (18, 249, 24), (173, 122, 5), (157, 162, 0), (147, 15, 212), (213, 167, 191),
                (99, 184, 147), (9, 233, 62), (87, 215, 183), (97, 20, 85), (102, 216, 48), (231, 175, 114),
                (143, 48, 180), (248, 5, 175), (161, 223, 71), (123, 125, 152), (247, 101, 95), (74, 170, 70),
                (199, 28, 93), (162, 158, 142), (21, 158, 205), (140, 111, 134), (113, 181, 61), (152, 68, 84),
                (82, 208, 49), (222, 95, 200), (239, 217, 190), (153, 112, 25), (83, 181, 69), (49, 116, 220),
                (89, 163, 167), (178, 109, 222), (25, 15, 190), (172, 117, 119), (72, 59, 73), (90, 244, 234),
                (125, 255, 27), (186, 97, 43), (159, 205, 79), (54, 88, 1), (135, 241, 119), (107, 251, 34),
                (147, 249, 248), (159, 102, 150), (208, 255, 88), (11, 98, 131), (77, 210, 227), (231, 1, 200),
                (19, 92, 218)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    @classmethod
    def fix_METAINFO_sentence(cls):
        cls.METAINFO = {'classes': cls.METAINFO['classes'][1:], 'palette': cls.METAINFO['palette'][1:]}

    def __init__(self, sentence: bool = False, **kwargs):
        self.sentence = sentence
        if not self.sentence:
            self.fix_METAINFO_sentence()
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`

        # 引入'sentence'类别
        if self.sentence:
            self.coco.dataset['categories'] = [{'id': 0, 'name': 'sentence', 'supercategory': 'char'}] + \
                                              self.coco.dataset['categories']
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            if 'sentence_id' in ann:
                instance['sentence_id'] = ann['sentence_id']

            if 'words_order' in ann:
                instance['words_order'] = ann['words_order']

            instance['instance_id'] = i

            instances.append(instance)

        if self.sentence:
            def merge_bboxes(bboxes):
                assert len(bboxes) > 0
                x_min = min(bbox[0] for bbox in bboxes)
                y_min = min(bbox[1] for bbox in bboxes)
                x_max = max(bbox[2] for bbox in bboxes)
                y_max = max(bbox[3] for bbox in bboxes)
                return [x_min, y_min, x_max, y_max]

            sentences_bboxes = {}

            for instance in instances:
                sentence_id = instance['sentence_id']
                if sentence_id not in sentences_bboxes:
                    sentences_bboxes[sentence_id] = []
                sentences_bboxes[sentence_id].append(instance['bbox'])

            sentence_instances = []

            for sentence_id, bboxes in sentences_bboxes.items():
                sentence_instance = {'bbox': merge_bboxes(bboxes), 'sentence_id': sentence_id, 'bbox_label': 0,
                                     'ignore_flag': 0}
                sentence_instances.append(sentence_instance)

            instances.extend(sentence_instances)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
