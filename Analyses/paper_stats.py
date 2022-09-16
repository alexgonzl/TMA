import numpy as np
import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

import ipywidgets as widgets
from ipywidgets import interact, fixed, interact_manual
from IPython.display import display

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class LMM_Stats():
    def __init__(self, info):
        self.info = info

    def segment_rates(self):
        """creates a model that test the hypothesis that there's a difference in the way segments are coded.
        return
        widget output.
        """

        seg_rates = self.info.get_segment_rate_comps()
        out = widgets.interactive(self._segment_rates, seg_rates=fixed(seg_rates),
                                  unit_type=['all', 'cell', 'mua'],
                                  metric_type=['uz_val', 't_val'],
                                  comp_type=['cue', 'rw', 'dir'])

        display(out)
        return out

    def remap_scores(self, **remap_params):
        scores = self.info.get_zone_rates_remap(overwrite=False, **remap_params)
        out = widgets.interactive(self._remap_scores, scores=fixed(scores),
                                  unit_type=['all', 'cell', 'mua'],
                                  metric_type=['zm', 'zt'],
                                  comp_type=['cue', 'rw']
                                  )

        display(out)
        return out

    def unit_remap_to_beh(self, **remap_params):
        """individual unit scores to behavior"""
        zrc = self.info.get_zone_rates_remap(overwrite=False, **remap_params)
        b_table = self.info.get_behav_perf()
        scores = self._combine_tables(zrc, b_table)

        out = widgets.interactive(self._unit_remap_beh, scores=fixed(scores), unit_type=['all', 'cell', 'mua'],
                                  metric_type=['zm', 'zt'],
                                  comp_type=['cue', 'rw'],
                                  beh_score=['pct_correct', 'pct_sw_correct'],
                                  rescale_behav=True,
                                  mean_unit_session=False,
                                  )

        display(out)
        return out

    def pop_remap_to_beh(self, **remap_params):
        """
        population remapping scores to behavior
        :param remap_params:
        :return:
        """
        zrc = self.info.get_pop_zone_rates_remap(overwrite=False, **remap_params)
        b_table = self.info.get_behav_perf()
        scores = self._combine_tables(zrc, b_table)

        out = widgets.interactive(self._pop_remap_beh, scores=fixed(scores), unit_type=['all', 'cells', 'muas'],
                                  metric_type=['zm', 'zt'],
                                  comp_type=['cue', 'rw'],
                                  beh_score=['pct_correct', 'pct_sw_correct'],
                                  rescale_behav=True,
                                  include_n_units=False,
                                  )

        display(out)
        return out

    def zone_encoder_scores(self):
        scores = self.info.get_zone_encoder_comps()

        out = widgets.interactive(self._zone_encoder_scores, scores=fixed(scores), unit_type=['all', 'cell', 'mua'],
                                  comp_type=['cue', 'rw', 'dir'],
                                  score_type=['mean_test', 'uz', 'md'],
                                  )

        display(out)
        return out

    def open_field_models(self):
        _, scores = self.info.get_of_results(model_analyses='new')

        out = widgets.interactive(self._open_field_models, scores=fixed(scores), unit_type=['all', 'cell', 'mua'],
                                  score_type=['r2', 'map_r'],
                                  test_model=['speed', 'hd', 'pos', 'agg_sdp'],
                                  joint_fit=False
                                  )

        display(out)
        return out

    def of_cluster_TM_comps(self):
        cluster_table = self.info.get_matched_of_cell_clusters()
        match_table = self.info.get_unit_match_table()
        cm_table = match_table[['match_cl_id', 'subject', 'session_T3', 'session_OF']].copy()
        cm_table[['cluster', 'umap_1', 'umap_2']] = cluster_table.loc[cm_table['match_cl_id'], ['Cluster', 'UMAP-1', 'UMAP-2']].values
        cm_table['cluster'] = cm_table.cluster.astype(pd.api.types.CategoricalDtype())
        scores = self.info.get_combined_scores_matched_units(mean_multi_matches=False)

        out = widgets.interactive(self._of_cluster_TM_comps, scores=fixed(scores), cm_table=fixed(cm_table),
                                  score_group=['of_coef', 'of_metrics', 'tm_remap', 'tm_enc_r2', 'tm_enc_delta'],
                                  cluster_type=['cluster', 'umap_1', 'umap_2'],
                                  test_type=['main', 'interaction'],
                                  )

        display(out)
        return out

    @staticmethod
    def _segment_rates(seg_rates, unit_type, metric_type, comp_type):
        """creates a model that test the hypothesis that there's a difference in the way segments are coded.
        :param: unit_type. str. ['all', 'cell', 'mua']
        :param: metric_type. str. ['uz_val', 't']
        :param: comp_type. str. ['cue', 'rw']
        """

        if unit_type == 'all':
            df = seg_rates[seg_rates.comp == comp_type].copy()
        elif unit_type in ['cell', 'mua']:
            df = seg_rates[(seg_rates.comp == comp_type) & (seg_rates.unit_type == unit_type)].copy()
        else:
            raise ValueError

        df['score'] = df[metric_type]

        vc_formula = {'task': f"1+C(task)",
                      'session': f"0+C(session)"}

        if unit_type == 'all':
            vc_formula['unit'] = f"0+C(unit_type)"
        full_formula = "score ~ 0 + segment"
        null_formula = "score ~ 1"

        m_full = smf.mixedlm(formula=full_formula,
                             groups='subject', re_formula="1", vc_formula=vc_formula,
                             data=df).fit()

        print(m_full.summary())
        print(m_full.wald_test_terms())
        print()

        m_full_ML = smf.mixedlm(formula=full_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False)
        m_null_ML = smf.mixedlm(formula=null_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False)

        lrt, chi2_p = LRT(m_full_ML, m_null_ML)
        print(f"LRT = {lrt:0.2f}; Chi2_p={chi2_p}")
        print(f"Full Model ML converged = {m_full_ML.converged}")
        print(f"Null Model ML converged = {m_null_ML.converged}")

        if comp_type == 'cue':

            print()
            print("Interactions")

            print('c0: left>stem')
            print('c1: stem>right')
            print('c2: right>left')
            r_mat = np.zeros((3, len(m_full.bse_fe)))
            r_mat[0, [0, 1]] = 1, -1
            r_mat[1, [1, 2]] = 1, -1
            r_mat[2, [0, 2]] = -1, 1

            interactions_res = m_full.t_test(r_mat)
            print(interactions_res)
            print(interactions_res.pvalue)

            print()
            print("Joint Hypothesis Test")
            print("Left>Stem>Right")
            r_mat_joint = np.zeros((3, len(m_full.params)))
            r_mat_joint[0, [0, 1]] = 1, -1
            r_mat_joint[1, [1, 2]] = -1, 1
            r_mat_joint[2, [0, 2]] = 1, -1

            print(m_full.wald_test(r_mat_joint))

        elif comp_type == 'rw':

            idx = (seg_rates.comp == comp_type) & (seg_rates.unit_type == unit_type)
            t = pd.crosstab(seg_rates.loc[idx, metric_type] <= 0, seg_rates.loc[idx, 'segment'])
            print(f"Proportion of {metric_type}<0 by segment.")
            print(t)
            print(t.sum())

        return m_full

    @staticmethod
    def _remap_scores(scores, unit_type, metric_type, comp_type):
        """creates a model that test the hypothesis that there's a difference in the way segments are coded.
                :param: unit_type. str. ['all', 'cell', 'mua']
                :param: metric_type. str. ['uz_val', 't']
                :param: comp_type. str. ['cue', 'rw']
                """

        if comp_type == 'cue':
            comp = 'CR_bo-CL_bo-Even_bo-Odd_bo-corr_' + metric_type
        elif comp_type == 'rw':
            comp = 'Co_bi-Inco_bi-Even_bi-Odd_bi-corr_' + metric_type
        else:
            raise ValueError

        if unit_type == 'all':
            df = scores.copy()
        elif unit_type in ['cell', 'mua']:
            df = scores[(scores.unit_type == unit_type)].copy()
        else:
            raise ValueError

        df['task'] = df.session.apply(lambda x: x.split('_')[1])
        df['score'] = df[comp]
        df = df[['subject', 'session', 'task', 'unit_type', 'score']]
        df.dropna(inplace=True)
        df = df.reset_index()

        vc_formula = {'task': "1+C(task)",
                      'session': "0+C(session)"}

        if unit_type == 'all':
            full_formula = "score ~ 1 + unit_type"
        else:
            full_formula = "score ~ 1"

        m_full = smf.mixedlm(formula=full_formula,
                             groups='subject', re_formula="1", vc_formula=vc_formula,
                             data=df).fit()

        print(m_full.summary())
        print(m_full.wald_test_terms())
        print()

        return m_full

    @staticmethod
    def _unit_remap_beh(scores, unit_type, metric_type, comp_type, beh_score, rescale_behav, mean_unit_session):
        if comp_type == 'cue':
            comp = 'CR_bo-CL_bo-Even_bo-Odd_bo-corr_' + metric_type
        elif comp_type == 'rw':
            comp = 'Co_bi-Inco_bi-Even_bi-Odd_bi-corr_' + metric_type
        else:
            raise ValueError

        if unit_type == 'all':
            df = scores.copy()
        elif unit_type in ['cell', 'mua']:
            df = scores[(scores.unit_type == unit_type)].copy()
        else:
            raise ValueError

        df['task'] = df.session.apply(lambda x: x.split('_')[1])
        df['remap'] = df[comp]
        df['behav'] = df[beh_score]
        df = df[['subject', 'session', 'task', 'unit_type', 'remap', 'behav']]
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        if mean_unit_session:
            df = df.groupby(['subject', 'task', 'session']).mean()
            df = df.reset_index()

        if rescale_behav:
            df['behav'] = logit(df['behav'])

        vc_formula = {'task': f"1+C(task)"}

        if (unit_type == 'all') & (mean_unit_session is False):
            vc_formula['unit'] = f"0+C(unit_type)"

        full_formula = "behav ~ 1 + remap"
        null_formula = "behav ~ 1"

        m_full = smf.mixedlm(formula=full_formula,
                             groups='subject', re_formula="1", vc_formula=vc_formula,
                             data=df).fit()

        print(m_full.summary())
        print(m_full.wald_test_terms())
        print()

        m_full_ML = smf.mixedlm(formula=full_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False)
        m_null_ML = smf.mixedlm(formula=null_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False)

        lrt, chi2_p = LRT(m_full_ML, m_null_ML)
        print(f"LRT = {lrt:0.2f}; Chi2_p={chi2_p}")
        print(f"Full Model ML converged = {m_full_ML.converged}")
        print(f"Null Model ML converged = {m_null_ML.converged}")

        return m_full

    @staticmethod
    def _pop_remap_beh(scores, unit_type, metric_type, comp_type, beh_score, rescale_behav, include_n_units):
        if comp_type == 'cue':
            comp = 'CR_bo-CL_bo-Even_bo-Odd_bo-corr_' + metric_type
        elif comp_type == 'rw':
            comp = 'Co_bi-Inco_bi-Even_bi-Odd_bi-corr_' + metric_type
        else:
            raise ValueError

        df = scores[scores.pop_type == unit_type].copy()

        df['task'] = df.session.apply(lambda x: x.split('_')[1])
        df['remap'] = df[comp]
        df['behav'] = df[beh_score]
        df['n_units'] = df[f"n_session_{unit_type}"]
        df = df[['subject', 'session', 'task', 'remap', 'behav', 'n_units']]
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        if rescale_behav:
            df['behav'] = logit(df['behav'])

        vc_formula = {'task': f"1+C(task)"}

        if include_n_units:
            re_formula = "1+n_units"
        else:
            re_formula = "1"
        full_formula = "behav ~ 1 + remap"
        null_formula = "behav ~ 1"

        m_full = smf.mixedlm(formula=full_formula,
                             groups='subject', re_formula=re_formula, vc_formula=vc_formula,
                             data=df).fit()

        print(m_full.summary())
        print(m_full.wald_test_terms())
        print()

        m_full_ML = smf.mixedlm(formula=full_formula,
                                groups='subject', re_formula=re_formula, vc_formula=vc_formula,
                                data=df).fit(reml=False)
        m_null_ML = smf.mixedlm(formula=null_formula,
                                groups='subject', re_formula=re_formula, vc_formula=vc_formula,
                                data=df).fit(reml=False)

        lrt, chi2_p = LRT(m_full_ML, m_null_ML)
        print(f"LRT = {lrt:0.2f}; Chi2_p={chi2_p}")
        print(f"Full Model ML converged = {m_full_ML.converged}")
        print(f"Null Model ML converged = {m_null_ML.converged}")

        return m_full

    @staticmethod
    def _zone_encoder_scores(scores, unit_type, comp_type, score_type):

        if unit_type == 'all':
            df = scores[scores.expt == comp_type].copy()
        elif unit_type in ['cell', 'mua']:
            df = scores[(scores.expt == comp_type) & (scores.unit_type == unit_type)].copy()
        else:
            raise ValueError

        # take out unstable scores
        df.loc[df.mean_test < -1, 'mean_test'] = np.nan
        # df.loc[df.mean_test < -2] = np.nan

        df['test_cond'] = df['test_cond'].astype(
            pd.api.types.CategoricalDtype(['none', 'fixed', 'inter']))
        df['score'] = df[score_type]

        df = df[['subject', 'session', 'task', 'score', 'test_cond', 'comp', 'unit_type']]
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        vc_formula = {
            'task': f"1+C(task)",
            'session': f"0+C(session)",
        }

        if unit_type == 'all':
            vc_formula['unit'] = f"0+C(unit_type)"

        if score_type == 'mean_test':
            full_formula = "score ~ 0 + test_cond"
        elif score_type in ['uz', 'md']:
            full_formula = "score ~ 0 + comp"
        else:
            raise ValueError
        null_formula = "score ~ 1"

        m_full = smf.mixedlm(formula=full_formula,
                             groups='subject', re_formula="1", vc_formula=vc_formula,
                             data=df).fit()

        print(m_full.summary())
        print(m_full.wald_test_terms())
        print()

        m_full_ML = smf.mixedlm(formula=full_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False, start_params=m_full.params)
        m_null_ML = smf.mixedlm(formula=null_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False)

        lrt, chi2_p = LRT(m_full_ML, m_null_ML)
        print(f"LRT = {lrt:0.2f}; Chi2_p={chi2_p}")
        print(f"Full Model ML converged = {m_full_ML.converged}")
        print(f"Null Model ML converged = {m_null_ML.converged}")

        return m_full

    @staticmethod
    def _open_field_models(scores, unit_type, score_type, test_model, joint_fit):
        exclude_units = scores[(scores.value <= -1) & (scores.metric == 'r2')].unit_id.unique()
        exclude_units = np.union1d(exclude_units, scores[scores.unit_type == 'mua'].unit_id.unique())
        valid_untis = np.setdiff1d(scores.unit_id.unique(), exclude_units)

        df = scores[scores.unit_id.isin(valid_untis)].copy()
        if unit_type != 'all':
            df = df[df.unit_type == unit_type]
        df = df.loc[(df.metric == score_type) & (df.split == 'test'),
                    ['unit_id', 'model', 'subject', 'session', 'split', 'value', 'unit_type']].copy()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        vc_formula = {'session': f"0+C(session)",
                      'unit_id': f"0+C(unit_id)"}
        if unit_type == 'all':
            vc_formula['unit'] = f"0+C(unit_type)"

        models = df.model.unique()
        for m in models:
            df[m] = 0
            df.loc[df.model == m, m] = 1

        full_formula = 'value ~ 0'
        null_formula = 'value ~ 0'

        if joint_fit:
            full_formula += '+ model'
            null_formula = 'value ~ 1'
        else:
            for m in models:
                full_formula += f'+ {m}'
                if m != test_model:
                    null_formula += f'+ {m}'

        m_full = smf.mixedlm(formula=full_formula,
                             groups='subject', re_formula="1", vc_formula=vc_formula,
                             data=df).fit()

        print(m_full.summary())
        print(m_full.wald_test_terms())
        print()

        m_full_ML = smf.mixedlm(formula=full_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False, start_params=m_full.params)
        m_null_ML = smf.mixedlm(formula=null_formula,
                                groups='subject', re_formula="1", vc_formula=vc_formula,
                                data=df).fit(reml=False)

        lrt, chi2_p = LRT(m_full_ML, m_null_ML)
        print(f"LRT = {lrt:0.2f}; Chi2_p={chi2_p:0.2e}")
        print(f"Full Model ML converged = {m_full_ML.converged}")
        print(f"Null Model ML converged = {m_null_ML.converged}")

        if not joint_fit:
            p = m_full.params.index
            A = np.zeros((len(p), len(p)))
            t_idx = np.nan
            for ii, m in enumerate(models):
                if m == test_model:
                    A[:, p == test_model] = 1
                    t_idx = ii
                if m != test_model:
                    A[ii, p == m] = -1
            A = A[:len(models)]
            A = np.delete(A, t_idx, 0)

            print()
            print(f"Joint Hypothesis Test: {test_model}!=all other models.")
            print()

            print()
            print(m_full.wald_test(A))
            print("Pairwise Comps")
            print(m_full.t_test(A[:, :len(models)]))

        return m_full

    @staticmethod
    def _of_cluster_TM_comps(scores, cm_table, score_group, cluster_type, test_type):

        of_vars = ['speed', 'hd', 'pos']
        if score_group == 'of_coef':
            score_columns = [f"OF-{m}-agg_sdp_coef" for m in of_vars]
            abbreviations = of_vars
        elif score_group == 'of_metrics':
            score_columns = [f'OF-metric_score_{m}' for m in of_vars]
            abbreviations = of_vars
        elif score_group == 'tm_remap':
            score_columns = ['TM-remap_cue', 'TM-remap_rw']
            abbreviations = ['cue', 'rw']
        elif score_group == 'tm_enc_r2':
            score_columns = ['TM-rate_cue', 'TM-global_cue', 'TM-rate_rw', 'TM-global_rw', ]
            abbreviations = [s.split('-')[1] for s in score_columns]
        elif score_group == 'tm_enc_delta':
            score_columns = ['TM-enc_uz_cue', 'TM-enc_uz_rw']
            abbreviations = ['cue', 'rw']
        else:
            raise ValueError

        df = scores[score_columns].copy()
        df[['match_cl_id', 'subject', 'session_TM', 'session_OF']] = cm_table[['match_cl_id', 'subject',
                                                                               'session_T3', 'session_OF']].copy()
        df[cluster_type] = cm_table[cluster_type].copy()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = df.rename(columns={b: a for a, b in zip(abbreviations, score_columns)})
        df = df.melt(id_vars=[cluster_type, 'subject', 'match_cl_id', 'session_TM', 'session_OF'], value_name='score',
                     var_name=score_group)

        vc = {'session_TM': f"0+C(session_TM)",
              'session_OF': f"0+C(session_OF)",
              'match_id': f"0+C(match_cl_id)"
              }

        if score_group == 'tm_enc_r2':
            cue_idx = df[score_group].apply(lambda x: (x.split('_')[1])) == 'cue'
            rate_idx = df[score_group].apply(lambda x: (x.split('_')[0])) == 'rate'
            df['cue'] = cue_idx
            df['rate'] = rate_idx

        if test_type == 'interaction':
            if score_group == 'tm_enc_r2':
                full_formula = f"score ~ 1 + rate + cue * {cluster_type}"
                null_formula = f"score ~ 1 + rate + cue + {cluster_type}"
            else:
                full_formula = f"score ~ 1 + {score_group} * {cluster_type}"
                null_formula = f"score ~ 1 + {score_group} + {cluster_type}"
        elif test_type == 'main':
            if score_group == 'tm_enc_r2':
                full_formula = f"score ~ 1 + rate + cue + {cluster_type}"
                null_formula = f"score ~ 1 + rate + cue"
            else:
                full_formula = f"score ~ 1 + {score_group} + {cluster_type}"
                null_formula = f"score ~ 1 + {score_group}"

        m_full = smf.mixedlm(formula=full_formula,
                             groups='subject', re_formula='1', vc_formula=vc, data=df).fit()
        print(m_full.summary())
        print(m_full.wald_test_terms())
        print()

        m_full_ML = smf.mixedlm(formula=full_formula,
                                groups='subject', re_formula="1", vc_formula=vc,
                                data=df).fit(reml=False, start_params=m_full.params)
        m_null_ML = smf.mixedlm(formula=null_formula,
                                groups='subject', re_formula="1", vc_formula=vc,
                                data=df).fit(reml=False)

        lrt, chi2_p = LRT(m_full_ML, m_null_ML)
        print(f"LRT = {lrt:0.2f}; Chi2_p={chi2_p:0.2e}")
        print(f"Full Model ML converged = {m_full_ML.converged}")
        print(f"Null Model ML converged = {m_null_ML.converged}")

        return m_full

    @staticmethod
    def _combine_tables(zrc, b_table):
        zrc_b = zrc.copy()
        b_table = b_table.copy()
        b_table.set_index('session', inplace=True)

        b_cols = ['pct_correct', 'pct_sw_correct', 'pct_vsw_correct', 'pct_L_correct', 'pct_R_correct']
        for session in b_table.index:
            z_index = zrc_b.session == session
            zrc_b.loc[z_index, b_cols] = b_table.loc[session, b_cols].values

        zrc_b['task'] = zrc_b.session.apply(lambda x: x.split('_')[1])

        return zrc_b


def LRT(full_fit, null_fit):
    """
    returns the log-likelihood ratio test and the resulting p value from statsmodels results
    :param full_fit:
    :param null_fit:
    :return:
        lrt value
        p_value
    """

    # delta_p = full_fit.df_model - null_fit.df_model
    delta_p = len(full_fit.params) - len(null_fit.params)
    lrt = 2 * (full_fit.llf - null_fit.llf)
    chi2_p = stats.chi2.sf(lrt, delta_p)
    return lrt, chi2_p


def logit(p):
    return np.log(p / (1 - p))
