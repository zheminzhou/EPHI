import subprocess, re
import sys, os, numpy as np, pandas as pd
import click, ete3

from ete3_extensions import read_nexus, prune, write_nexus
from scipy.stats import linregress
from multiprocessing import Pool
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=20)

bin_dir = os.path.dirname(sys.executable)
Rscript = 'Rscript' if not os.path.isfile(os.path.join(bin_dir, 'Rscript')) \
    else os.path.join(bin_dir, 'Rscript')
treetime = 'treetime' if not os.path.isfile(os.path.join(bin_dir, 'treetime')) \
    else os.path.join(bin_dir, 'treetime')


def extract_dated_tree(tre, date_file):
    dates = {}
    with open(date_file) as fin:
        for line in fin:
            p = line.strip().split('\t')
            try:
                if len(p) >= 3:
                    s, e = float(p[1]), float(p[2])
                elif len(p) == 2:
                    s, e = float(p[1]), float(p[1])
                dates[p[0]] = [s, e]
            except:
                logging.info('LINE ignored: {0}'.format(line))

    tre = prune(tre, list(dates.keys()))
    #tre = resolve_polytomy(tre, 1e-8, 2)
    tre.resolve_polytomy(1e-8)

    n_names = {n.name: 1 for n in tre.traverse()}
    for id, node in enumerate(tre.traverse('postorder')):
        if not node.name:
            n = 'NODE_{:0>6}'.format(id)
            node.name, i = n, 1
            while node.name in n_names:
                node.name = n + '_{0}'.format(i)
                i += 1
        if 'annotations' not in node.__dict__:
            node.annotations = {}
        if node.is_leaf():
            node.annotations['date'] = dates[node.name]
    return tre


def find_dated_nodes(tre, min_r, max_p):
    tre.up = None
    for node in tre.traverse('preorder'):
        node.used = False
    logging.info('#NODE\tN_tips\tR_value\tRegression\tPossibility')
    for node in tre.traverse('preorder'):
        if node.up and node.up.used:
            node.used = True
            continue
        if node.annotations['r'] >= min_r and node.annotations['p'] <= max_p:
            logging.info('{0}\t{1}\t{r:.04f}\t{interception:.08f}+{slope:.08f}*year\t{p}'.format(node.name,
                                                                                                 len(node.get_leaves()),
                                                                                                 **node.annotations))
            node.used = True
            node.annotations['dated_node'] = 1
        elif node.is_root():
            logging.info('{0}\t{1}\t{r:.04f}*\t{interception:.08f}+{slope:.08f}*year\t{p}'.format(node.name,
                                                                                                  len(node.get_leaves()),
                                                                                                  **node.annotations))


def get_descendants_for_nodes(tre):
    for node in tre.traverse('postorder'):
        if node.is_leaf():
            node.d = {node.name: [node.annotations['date'][1], 0.]}
        else:
            node.d = {t: [d, l + c.dist] for c in node.children for t, (d, l) in c.d.items()}


def prepare(outdir, fname, default_fn):
    if not os.path.isdir(outdir):
        logging.info('The outdir does not exist. Create a new folder.')
        os.makedirs(outdir)
    if (not fname) and default_fn:
        logging.info('Input file is not specified. Use default file "{0}"'.format(os.path.join(outdir, default_fn)))
        fname = os.path.join(outdir, default_fn)
    return fname


def rtt(dat, outlier=False):
    r = linregress(dat.T[0], dat.T[1])
    if outlier:
        return r, []
    dist = (dat.T[0] * r[0] + r[1]) - dat.T[1]
    q75, q25 = np.quantile(dist, 0.75), np.quantile(dist, 0.25)
    idx = (dist >= q25 - 3 * (q75 - q25)) & (dist <= q75 + 3 * (q75 - q25))
    if np.unique(dat[idx, 0]).size > 1:
        r2 = linregress(dat[idx, 0], dat[idx, 1])
        return (r2, np.where(~idx)[0]) if r2[2] > r[2] else (r, np.where(~idx)[0])
    else:
        return (r, np.where(~idx)[0])


@click.group()
@click.option('-n', '--n_threads', help='number of threads. default: 8', default=8, type=int)
def cli(n_threads):
    global pool
    pool = Pool(n_threads)
    logging.info('CMD: {0}'.format(' '.join(sys.argv)))


def eval_reroot(data):
    tre, node = data
    tre.set_outgroup(node)
    get_descendants_for_nodes(tre)

    dist_sum = np.sum([c.dist for c in tre.children])
    scope = [0., dist_sum]
    for _ in np.arange(2):
        trials = np.linspace(scope[0], scope[1], 7)
        rs = []
        for d0 in trials:
            tre.children[0].dist = d0
            tre.children[1].dist = dist_sum - d0
            dat = np.array([[d, l + c.dist] for c in tre.children for t, (d, l) in c.d.items()])
            r, _ = rtt(dat)
            rs.append(r[2])
        idx = np.argmax(rs)
        result = rs[idx]
        scope = [trials[max(idx - 1, 0)], trials[min(idx + 1, 6)]]
    return result, node


@cli.command()
@click.option('-t', '--tree', help='newick tree scaled in mutation per site')
@click.option('-d', '--date', help='file storing isolation date')
@click.option('-o', '--outdir', help='dirname that is used to store generated files.', default='evol_dir')
def reroot(tree, date, outdir):
    _reroot(tree, date, outdir)


def _reroot(tree, date, outdir):
    prepare(outdir, tree, '')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    tre = ete3.Tree(tree, format=1)
    tre = extract_dated_tree(tre, date)

    nodes = tre.get_descendants()
    node_rs = list(pool.map(eval_reroot, [[tre, node] for node in nodes]))
    s, node = max(node_rs, key=lambda x: x[0])
    tre.set_outgroup(tre.search_nodes(name=node.name)[0])
    get_descendants_for_nodes(tre)

    dist_sum = np.sum([c.dist for c in tre.children])
    scope = [0., dist_sum]
    for _ in np.arange(5):
        trials = np.linspace(scope[0], scope[1], 7)
        rs = []
        for d0 in trials:
            tre.children[0].dist = d0
            tre.children[1].dist = dist_sum - d0
            dat = np.array([[d, l + c.dist] for c in tre.children for t, (d, l) in c.d.items()])
            r, _ = rtt(dat)
            rs.append(r[2])
        idx = np.argmax(rs)
        scope = [trials[max(idx - 1, 0)], trials[min(idx + 1, 6)]]
    tre.children[0].dist = trials[idx]
    tre.children[1].dist = dist_sum - trials[idx]

    with open(os.path.join(outdir, 'root.nwk'), 'wt') as fout:
        fout.write(tre.write(format=1).rsplit(')', 1)[0] + ')' + tre.name + ';')
    return os.path.join(outdir, 'root.nwk')


@cli.command()
@click.option('-o', '--outdir', help='dirname that is used to store generated files.', default='evol_dir')
@click.option('-t', '--tree', help='[used in signal] newick tree scaled in mutation per site')
@click.option('-d', '--date', help='[used in signal] file storing isolation date')
@click.option('-N', '--node', help='[used in dating] run on a subtree that rooted from a specified node', default=None)
@click.option('-r', '--reroot', help='flag to try to reroot the tree', default=False, is_flag=True)
@click.option('-l', '--n_aln', help='[used in dating] size of alignment in bps', type=int, required=True)
@click.option('-c', '--n_chain',
              help='[used in dating] total length of the MCMC chain [default:5e6]. Note that only last half the the chain will be used for posterior distribution',
              type=int, default=5e6)
@click.option('-m', '--model',
              help='model of substitutions. default: "strictgamma,mixedgamma,carc" allow combinations of strictgamma [strict clock], mixedgamma [relaxed clock], carc, arc, poisson, negbin, relaxedgamma, null, mixedcarc, poissonR, negbinR, strictgammaR, relaxedgammaR, arcR, carcR.',
              default='strictgamma,mixedgamma,carc')
@click.option('-R', '--rechmm', help='[used in dating] recombination.region file generated by RecHMM', default=None)
@click.option('-p', '--n_pop', help='[used in popsize] number of changes of effective population sizes. [default:20]',
              type=int, default=20)
@click.option('-M', '--map',
              help='[used in popsize] flag to use MAP method for effective population (faster, less accurate).',
              is_flag=True, default=False)
@click.option('-s', '--state',
              help='[used in trait] state file containing trait information. Add to trigger trait function')
def all(outdir, tree, date, node, reroot, n_aln, n_chain, model, n_pop, map, state, rechmm):
    if reroot:
        tree = _reroot(tree, date, outdir)
    _signal(tree, date, outdir)
    mutation_rate = _dating(None, outdir, n_chain, n_aln, node, model, reroot, rechmm)
    if state:
        _trait(None, outdir, None, state, mutation_rate)
    _popsize(None, outdir, map, n_pop, None)


def _node_rtt(data):
    node, min_n, min_r = data
    res = [0., 0., 0., 1., np.zeros(0)]
    if len(node.d) >= min_n:
        d = list(node.d.items())
        dat = np.array([x[1] for x in d])
        tips = np.array([x[0] for x in d])
        if len(np.unique(dat.T[0])) > 1:
            r0, outliers = rtt(dat)
            res[4] = tips[outliers]
            res[:3] = r0[:3]
            if r0[2] > min_r or node.is_root():
                r1 = linregress(dat.T[0], dat.T[1])
                r_cnt = 0
                for _ in np.arange(1000):
                    r2 = linregress(dat.T[0], np.random.permutation(dat.T[1]))
                    if r2[2] >= r1[2]:
                        r_cnt += 1
                res[3] = r_cnt / 1000.
    return res


def resolve_polytomy(tre, min_dist, n_polytomy):
    for node in tre.traverse():
        while len(node.children) > n_polytomy:
            n_children = len(node.children)
            children = np.random.choice(node.children, n_children, replace=False)
            node.children = []
            n = max(int(n_children / np.ceil(n_children / float(n_polytomy)) + 0.5), 2)
            for i in np.arange(0, n_children, n):
                nc = children[i:i + n]
                nn = ete3.TreeNode(dist=min_dist)
                nn.up = node
                for c in nc:
                    nn.add_child(c)
                    c.up = nn
                node.add_child(nn)
    return tre


@cli.command()
@click.option('-t', '--tree', help='newick tree scaled in mutation per site')
@click.option('-d', '--date', help='file storing isolation date')
@click.option('-o', '--outdir', help='dirname that is used to store generated files.', default='evol_dir')
@click.option('-n', '--min_n', help='minimum number of descendant tips for a date node [default: 8]', default=8)
@click.option('-r', '--min_r', help='minimum r value for a date node [default: 0.25]', default=0.25)
def signal(tree, date, outdir, min_n, min_r):
    _signal(tree, date, outdir, min_n, min_r)


def _signal(tree, date, outdir, min_n=8, min_r=0.25):
    prepare(outdir, tree, '')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    tre = ete3.Tree(tree, format=1)
    tre = extract_dated_tree(tre, date)

    get_descendants_for_nodes(tre)
    results = list(pool.map(_node_rtt, [[node, min_n, min_r] for node in tre.traverse('postorder')]))
    for node, res in zip(tre.traverse('postorder'), results):
        node.annotations['slope'] = res[0]
        node.annotations['interception'] = res[1]
        node.annotations['r'] = res[2]
        node.annotations['p'] = res[3]
        node.annotations['outliers'] = res[4].tolist()

    find_dated_nodes(tre, min_r, 0.01)
    with open(os.path.join(outdir, 'roottotip_signal.out.nex'), 'wt') as fout:
        fout.write(write_nexus([tre]))
    logging.info('Dating signal (root-to-tip regression) is saved in "{0}"'.format(
        os.path.join(outdir, 'roottotip_signal.out.nex')))


@cli.command()
@click.option('-n', '--nexus', help='nexus tree generated in signal', default=None)
@click.option('-o', '--outdir', help='dirname that is used to store generated files.', default='evol_dir')
@click.option('-l', '--n_aln', help='size of alignment in bps', type=int, required=True)
@click.option('-c', '--n_chain',
              help='total length of the MCMC chain [default:5e6]. Note that only last half the the chain will be used for posterior distribution',
              type=int, default=5e6)
@click.option('-N', '--node', help='run on a subtree that rooted from a specified node', default=None)
@click.option('-m', '--model',
              help='model of substitutions. default: "strictgamma,mixedgamma,carc" allow combinations of strictgamma [strict clock], mixedgamma [relaxed clock], carc, arc, poisson, negbin, relaxedgamma, null, mixedcarc, poissonR, negbinR, strictgammaR, relaxedgammaR, arcR, carcR.',
              default='strictgamma,mixedgamma,carc')
@click.option('-r', '--reroot', help='flag to try to reroot the tree', default=False, is_flag=True)
@click.option('-R', '--rechmm', help='recombination.region file generated by RecHMM', default=None)
def dating(nexus, outdir, n_chain, n_aln, node, model, reroot, rechmm):
    _dating(nexus, outdir, n_chain, n_aln, node, model, reroot, rechmm)


def runR(data):
    Rscript, i, outdir = data
    p = subprocess.Popen('{0} m{1}.rmd'.format(Rscript, i).split(), cwd=outdir, universal_newlines=True,
                         stdout=subprocess.PIPE)
    logging.info('')
    dic = 0.
    for line in p.stdout:
        sys.stdout.write(line.strip() + '\r')
        sys.stdout.flush()
        if line.find('DIC') >= 0:
            dic = float(re.findall('DIC=([\d\.-]+)', line)[0])
    return i, dic


def _dating(nexus, outdir, n_chain, n_aln, node, model, reroot, rechmm):
    nexus = prepare(outdir, nexus, 'roottotip_signal.out.nex')
    n_aln = float(n_aln)
    tre = read_nexus(nexus)[0]
    if node:
        for n in tre.traverse():
            if n.name == node:
                tre = n
                break
        else:
            raise ("node is not found.")

    for n in tre.traverse():
        n.dist = max(n.dist*n_aln, 1e-8)
        n.n_aln = 1.

    rec = {}
    if rechmm:
        n_aln = 0.
        with open(rechmm, 'rt') as fin:
            for line in fin:
                if line.startswith('Branch'):
                    p = line.strip().split('\t')
                    br, mu, size = p[1], float(p[2][2:]), float(p[4][2:])
                    rec[br] = [mu * size, size]
                    if size > n_aln:
                        n_aln = size
        for n in tre.traverse():
            if n.name in rec:
                n.dist = rec[n.name][0]
                n.n_aln = rec[n.name][1] / float(n_aln)

        with open(os.path.join(outdir, 'dating.rec'), 'wt') as fout:
            fout.write('Node,Rec\n')
            for n in tre.get_descendants():
                fout.write('{0},{1}\n'.format(n.name, n.n_aln))

    with open(os.path.join(outdir, 'dating.tre'), 'wt') as fout:
        fout.write(tre.write(format=1).rsplit(')', 1)[0] + '){0};'.format(tre.name))
    with open(os.path.join(outdir, 'dating.dates'), 'wt') as fout:
        for n in tre.get_leaves():
            fout.write('{0},{1}\n'.format(*n.annotations['date']))

    models = model.split(',')
    for i, m in enumerate(models):
        if not rec:
            main_block = "res = bactdate(tre,date=date, model='{1}', nbIts={0}, updateRoot={2}, showProgress=T)".format(
                n_chain, m, 'T' if reroot else 'F')
        else:
            main_block = '''
                edge_node <- as.matrix(c(tre$tip.label, tre$node.label)[tre$edge[,2]])
                colnames(edge_node) <- c('Node')
                node_rec <- read.csv('dating.rec')
                tre$unrec <- merge(edge_node, node_rec, by='Node', sort=F)[, 'Rec']
                res = bactdate(tre,date=date, model='{1}', nbIts={0}, updateRoot={2}, showProgress=T, useRec=T)
            '''.format(n_chain, m, 'T' if reroot else 'F')

        with open(os.path.join(outdir, 'm{0}.rmd'.format(i)), 'wt') as fout:
            fout.write('''
                require(BactDating)
                require(ape)
                require(coda)

                tre = read.tree(file='dating.tre')
                date = as.matrix(read.csv("dating.dates", header=F))
                {main_block}
                m{0} <- res
                save(m{0}, file='m{0}.RData')
                modelcompare(m{0}, m{0})

                ess <- effectiveSize(as.mcmc.resBactDating(res))
                write.csv(ess, "m{0}.ess", row.names=T)
                write.tree(phy=res$tree, file='m{0}.tre')
                write.csv(res$record, "m{0}.records", row.names=F)
                cat(res$tree$tip.label,res$tree$node.label, file='m{0}.nodes')
            '''.format(i, main_block=main_block))
    res = []
    for i, dic in pool.imap_unordered(runR, [[Rscript, i, outdir] for i, m in enumerate(models)]):
        res.append([dic, i])
    logging.info('')
    for dic, i in res:
        logging.info('Model: {0}; DIC: {1}'.format(models[i], dic))
    best_dic, best_model = max(res)

    with open(os.path.join(outdir, 'm{0}.nodes'.format(best_model))) as fin:
        node_names = np.array([p for line in fin.readlines() for p in line.strip().split()])

    ess = dict(pd.read_csv(os.path.join(outdir, 'm{0}.ess'.format(best_model))).values.tolist())
    dat = pd.read_csv(os.path.join(outdir, 'm{0}.records'.format(best_model)))
    colnames = dat.columns
    dat = dat.values[-int(0.5 * dat.shape[0]):]
    ave_dates = np.mean(dat.T[:node_names.size, :], 1)
    ci95 = np.vstack([np.percentile(dat.T[:node_names.size], q=2.5, axis=1),
                      np.percentile(dat.T[:node_names.size], q=97.5, axis=1)]).T
    ave_rates = np.mean(dat.T[2 * node_names.size:3 * node_names.size, :], 1)
    node_dates = {n: [m, ci.tolist(), r] for n, m, ci, r in zip(node_names, ave_dates, ci95, ave_rates)}

    tre = ete3.Tree(os.path.join(outdir, 'm{0}.tre'.format(best_model)), format=1)
    for node in tre.traverse('postorder'):
        node.annotations = {}
        node.annotations['date'] = node_dates[node.name][0]
        node.annotations['CI95'] = node_dates[node.name][1]
        node.annotations['mut.rate'] = node_dates[node.name][2] / (rec.get(node.name, [1., n_aln])[1])
    for node in tre.traverse('postorder'):
        if node.up:
            node.dist = node.annotations['date'] - node.up.annotations['date']

    colnames = colnames[3 * node_names.size:]
    dat = dat[:, 3 * node_names.size:]
    ave_val = np.mean(dat.T, 1)
    ci95_val = np.vstack([np.percentile(dat.T, q=2.5, axis=1), np.percentile(dat.T, q=97.5, axis=1)]).T
    logging.info('\nBest model: {0}; DIC: {1}'.format(models[best_model], best_dic))
    logging.info('#Key\tmean\t2.5%\t97.5%\tESS')
    for c, v, ci in zip(colnames, ave_val, ci95_val):
        if c == 'mu':
            v, ci[0], ci[1] = v / n_aln, ci[0] / n_aln, ci[1] / n_aln
            mutation_rate = v
        logging.info('{0}\t{1}\t{2}\t{3}\t{4}'.format(c, v, ci[0], ci[1], ess.get(c, '-')))

    with open(os.path.join(outdir, 'dating.out.nex'), 'wt') as fout:
        fout.write(write_nexus([tre]))
    logging.info('Dating result is saved in "{0}"'.format(os.path.join(outdir, 'dating.out.nex')))
    return mutation_rate


@cli.command()
@click.option('-n', '--nexus', help='nexus tree generated in dating', default=None)
@click.option('-o', '--outdir', help='dirname that is used to store generated files.', default='evol_dir')
@click.option('-N', '--node', help='run on a subtree that rooted from a specified node', default=None)
@click.option('-s', '--state', help='state file containing trait information')
@click.option('-m', '--mutation_rate', help='state file containing trait information', default=1e-7, type=float)
@click.option('--no_missing', help='remove tips with missing data', default=False, is_flag=True)
def trait(nexus, outdir, node, state, mutation_rate, no_missing):
    _trait(nexus, outdir, node, state, mutation_rate, no_missing)


def _trait(nexus, outdir, node, state, mutation_rate, no_missing=False):
    nexus = prepare(outdir, nexus, 'dating.out.nex')

    tre = read_nexus(nexus)[0]
    if node:
        for n in tre.traverse():
            if n.name == node:
                tre = n
                break
        else:
            raise ("node is not found.")

    states = {}
    with open(state) as fin:
        for line in fin:
            p = line.strip().split('\t')
            if len(p) > 1 and p[1]:
                states[p[0]] = p[1]
    with open(os.path.join(outdir, 'trait.states'), 'wt') as fout:
        fout.write('ID,state\n')
        for n in tre.get_leaf_names():
            fout.write('{0},{1}\n'.format(n, states.get(n, '?')))

    if no_missing:
        kept = [leaf for leaf in tre.get_leaf_names() if leaf in states]
        tre = prune(tre, kept)

    for node in tre.traverse('postorder'):
        node.dist *= mutation_rate
    with open(os.path.join(outdir, 'trait.nwk'), 'wt') as fout:
        fout.write(tre.write(format=1).rsplit(')', 1)[0] + ')' + tre.name + ';')

    subprocess.Popen(
        '{0} mugration --states trait.states --tree trait.nwk --confidence --missing-data ? --out trait.out'.format(
            treetime
        ).split(), cwd=outdir, env=os.environ.copy()).communicate()

    trait_conf = pd.read_csv(os.path.join(outdir, 'trait.out', 'confidence.csv'))
    chr_map = {}
    with open(os.path.join(outdir, 'trait.out', 'GTR.txt'), 'rt') as fin:
        fin.readline()
        for line in fin:
            p = line.strip().split(':')
            if len(p) > 1:
                chr_map[p[0]] = p[1].strip()
            else:
                break

    trait_conf.columns = [chr_map.get(c.strip(), c.strip()) for c in trait_conf.columns]
    node_map = {}

    for c in trait_conf.values:
        node_map[c[0]] = [[], []]
        for tr, n in zip(trait_conf.columns[1:], c[1:]):
            if n > 1e-4:
                node_map[c[0]][0].append(tr)
                node_map[c[0]][1].append(n)

    # trait_tre = read_nexus(os.path.join(outdir, 'trait.out', 'annotated_tree.nexus'))[0]
    for node in tre.traverse():
        node.dist /= mutation_rate
        node.annotations['traits.list'], node.annotations['traits.prop'] = node_map[node.name]
        node.annotations['state'] = node.annotations['traits.list'][np.argmax(node.annotations['traits.prop'])]
        node.annotations['state.prop'] = np.max(node.annotations['traits.prop'])
    with open(os.path.join(outdir, 'trait.out.nex'), 'wt') as fout:
        fout.write(write_nexus([tre]))
    logging.info('Trait mugration result is saved in "{0}"'.format(os.path.join(outdir, 'trait.out.nex')))


@cli.command()
@click.option('-n', '--nexus', help='nexus tree generated in dating', default=None)
@click.option('-o', '--outdir', help='dirname that is used to store generated files.', default='evol_dir')
@click.option('-N', '--node', help='run on a subtree that rooted from a specified node', default=None)
@click.option('-m', '--map', help='flag to use MAP method (faster, less accurate).', is_flag=True, default=False)
@click.option('-p', '--n_pop', help='number of changes of effective population sizes. [default:20]', type=int,
              default=20)
def popsize(nexus, outdir, map, n_pop, node):
    _popsize(nexus, outdir, map, n_pop, node)


def _popsize(nexus, outdir, map, n_pop, node):
    nexus = prepare(outdir, nexus, 'dating.out.nex')

    tre = read_nexus(nexus)[0]
    if node:
        for n in tre.traverse():
            if n.name == node:
                tre = n
                break
        else:
            raise ("node is not found.")
    tre.write(outfile=os.path.join(outdir, 'popsize.nwk'), format=1)

    with open(os.path.join(outdir, 'popsize.rmd'), 'wt') as fout:
        fout.write('''
            require(skygrowth)
            require(ape)

            tre <- read.tree(file='popsize.nwk')
            r <- skygrowth.{1}(tre, res = {0}, quiet=T, tau0=1.)
            {2}
            r$growthrate[length(r$growthrate)] = 0
            write.csv(cbind(r$time, r$ne_ci, r$growthrate), "popsize.list", row.names=FALSE)
            '''.format(n_pop, 'map' if map else 'mcmc', '' if map else 'r$growthrate <- r$growthrate_ci[, 2]'))
    subprocess.Popen('Rscript popsize.rmd'.split(), cwd=outdir).communicate()
    pop_sizes = pd.read_csv(os.path.join(outdir, 'popsize.list')).values
    cdate = max([t.annotations['date'] for t in tre.iter_leaves()])
    pop_sizes.T[0] = pop_sizes.T[0] - pop_sizes[-1, 0] + cdate
    pop_sizes = pd.DataFrame(pop_sizes, columns=['year', 'Ne_2.5%', 'Ne_meidan', 'Ne_97.5%', 'growth_rate'])
    pop_sizes.to_csv(os.path.join(outdir, 'popsize.out.csv'), index=False)
    logging.info(
        'Variation of effective population size is saved in "{0}"'.format(os.path.join(outdir, 'popsize.out.csv')))


if __name__ == '__main__':
    pool = None
    cli()
