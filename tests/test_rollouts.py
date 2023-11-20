import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass

from madrona_learn import (
    init,
    ActorCritic, RecurrentBackboneEncoder, BackboneShared,
    TrainConfig, PBTConfig,
)

init(0.5)

from madrona_learn.train_state import (
    PolicyState, PolicyTrainState, TrainStateManager, ObsPreprocessNoop,
)

from madrona_learn.rollouts import (
    RolloutConfig,
    RolloutState,
    RolloutManager,
    rollout_loop,
    _init_matchmake_assignments,
    _compute_reorder_chunks,
    _compute_reorder_state,
)

from madrona_learn.metrics import TrainingMetrics
from madrona_learn.moving_avg import EMANormalizer

def check_reorder_chunks(arr, P, C):
    B = arr.size // C + P - 1

    @jax.jit
    def reorder(arr):
        return _compute_reorder_chunks(arr, P, C, B)

    to_policy_idxs, to_sim_idxs = reorder(arr)

    policy_batches = jnp.take(
        arr, to_policy_idxs, mode='fill', fill_value=-1)

    assert jnp.sum(jnp.where(policy_batches != -1, 1, 0))
    arr_reconstructed = jnp.take(
        policy_batches.reshape(-1), to_sim_idxs, 0)

    #print(arr)
    #print(arr_reconstructed)
    #print(policy_batches)

    assert jnp.all(jnp.equal(arr, arr_reconstructed))

def test_reorder_chunks1():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 5, 3, 2, 1, 0, 3, 3])
    check_reorder_chunks(arr, P, C)

def test_reorder_chunks2():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 4, 5, 2, 1, 0, 3])
    check_reorder_chunks(arr, P, C)

def test_reorder_chunks3():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 4, 3, 2, 1, 0, 3])
    check_reorder_chunks(arr, P, C)

def test_reorder_chunks4():
    P = 6
    C = 4
    arr = jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
    arr = random.permutation(random.PRNGKey(5), arr, independent=True)
    check_reorder_chunks(arr, P, C)


def check_reorder(
    arr,
    num_current_policies,
    num_past_policies,
    policy_chunk_size_override,
):
    rollout_cfg = RolloutConfig.setup(
        num_current_policies = num_current_policies,
        num_past_policies = num_past_policies,
        num_teams = 2,
        team_size = 1,
        sim_batch_size = arr.size,
        self_play_portion = 0.0,
        cross_play_portion = 1.0,
        past_play_portion = 0.0,
        float_dtype = jnp.float16,
        policy_chunk_size_override = policy_chunk_size_override,
    )

    @jax.jit
    def reorder(arr):
        return _compute_reorder_state(arr, rollout_cfg)

    reorder_state = reorder(arr)

    policy_batches = jnp.take(
        arr, reorder_state.to_policy_idxs, mode='fill', fill_value=-1)

    assert jnp.sum(jnp.where(policy_batches != -1, 1, 0))
    arr_reconstructed = jnp.take(
        policy_batches.reshape(-1), reorder_state.to_sim_idxs, 0)

    #print(arr)
    #print(arr_reconstructed)
    #print(policy_batches)

    assert jnp.all(jnp.equal(arr, arr_reconstructed))

def test_reorder1():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 5, 3, 2, 1, 0, 3])
    check_reorder(arr, P, 0, C)


def setup_init_matchmake(
    num_current_policies,
    num_past_policies,
    num_teams,
    team_size,
    batch_size,
    self_play,
    cross_play,
    past_play,
    policy_chunk_size_override = 0
): 
    rollout_cfg = RolloutConfig.setup(
        num_current_policies = num_current_policies,
        num_past_policies = num_past_policies,
        num_teams = num_teams,
        team_size = team_size,
        sim_batch_size = batch_size,
        self_play_portion = self_play,
        cross_play_portion = cross_play,
        past_play_portion = past_play,
        float_dtype = jnp.float16,
        policy_chunk_size_override = policy_chunk_size_override,
    )

    @jax.jit
    def init_matchmake(rnd):
        return _init_matchmake_assignments(rnd, rollout_cfg)

    return init_matchmake(random.PRNGKey(7)).reshape(-1, num_teams, team_size)

def test_init_matchmake1():
    matchmake = setup_init_matchmake(
        num_current_policies = 4,
        num_past_policies = 0,
        num_teams = 1,
        team_size = 4,
        batch_size = 512,
        self_play = 1.0,
        cross_play = 0.0,
        past_play = 0.0,
    )

    print(matchmake)

def test_init_matchmake2():
    matchmake = setup_init_matchmake(
        num_current_policies = 4,
        num_past_policies = 3,
        num_teams = 2,
        team_size = 2,
        batch_size = 32,
        self_play = 0.0,
        cross_play = 0.5,
        past_play = 0.5,
    )

    print(matchmake[:4])
    print(matchmake[4:])


@dataclass(frozen=True)
class FakeActionDist:
    action: jax.Array

    def best(self):
        return self.action

    def sample(self, prng_key):
        return self.action, self.action


class FakeNet(nn.Module):
    @nn.compact
    def __call__(self, inputs, train):
        bias = self.param('bias',
            jax.nn.initializers.constant(0), (), jnp.int32)

        return jnp.concatenate([
            inputs + bias,
            jnp.broadcast_to(bias[None, None], inputs.shape),
        ], axis=-1)


class FakeRNN(nn.Module):
    @nn.nowrap
    def init_recurrent_state(self, N):
        return jnp.zeros((N, 1), dtype=jnp.int32)

    @nn.nowrap
    def clear_recurrent_state(self, rnn_states, should_clear):
        return jnp.where(
            should_clear, jnp.zeros((), dtype=jnp.int32), rnn_states)

    @nn.compact
    def __call__(self, cur_hiddens, in_features, train):
        y = in_features[..., 0:1] + cur_hiddens
        new_hiddens = cur_hiddens + 2 * in_features[..., 0:1]

        y = jnp.concatenate([y, in_features[..., 1:2], new_hiddens], axis=-1)

        return y, new_hiddens

def fake_rollout_setup(
    rnd,
    num_steps,
    episode_len,
    num_current_policies,
    num_past_policies,
    num_teams,
    team_size,
    batch_size,
    self_play,
    cross_play,
    past_play,
    policy_chunk_size_override = 0,
):
    rnd = random.PRNGKey(rnd)

    rollout_cfg = RolloutConfig.setup(
        num_current_policies = num_current_policies,
        num_past_policies = num_past_policies,
        num_teams = num_teams,
        team_size = team_size,
        sim_batch_size = batch_size,
        self_play_portion = self_play,
        cross_play_portion = cross_play,
        past_play_portion = past_play,
        float_dtype = jnp.int32,
        policy_chunk_size_override = policy_chunk_size_override,
    )

    rnd, rnd_obs = random.split(rnd)

    init_obs = random.randint(rnd_obs, (batch_size, 1), 0, 10000)

    init_sim_data = frozen_dict.freeze({
        'obs': init_obs,
        'rewards': jnp.zeros((batch_size, 1), dtype=jnp.int32),
        'dones': jnp.zeros((batch_size, 1), dtype=jnp.bool_),
        'actions': jnp.zeros((batch_size, 2), dtype=jnp.int32),
        'counter': jnp.zeros((batch_size, 1), dtype=jnp.int32),
    })

    def fake_sim(sim_data):
        sim_data = jax.tree_map(jnp.copy, sim_data)

        new_counter = sim_data['counter'] + 1
        new_dones = new_counter == episode_len

        new_counter %= episode_len

        return sim_data.copy({
            'obs': sim_data['actions'][..., 0:1] + 1,
            'rewards': sim_data['actions'][..., 0:1] + 2,
            'counter': new_counter,
            'dones': new_dones,
        })

    fake_backbone = BackboneShared(
        prefix = lambda x, train: x,
        encoder = RecurrentBackboneEncoder(
            net = FakeNet(),
            rnn = FakeRNN(),
        ),
    )

    def fake_actor(features, train):
        return FakeActionDist(features[..., 0:2])

    def fake_critic(features, train):
        return features[..., 2:3] + 1

    policy = ActorCritic(
        backbone = fake_backbone,
        actor = fake_actor,
        critic = fake_critic,
    )

    rnd, rnd_rollout, rnd_rnn = random.split(rnd, 3)

    @jax.jit
    def init_rollout_state():
        rnn_states = policy.init_recurrent_state(batch_size)

        rnn_states = jax.tree_map(
            lambda x: random.randint(rnd_rnn, x.shape, 0, 10000), rnn_states)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            step_fn = fake_sim,
            prng_key = rnd_rollout,
            rnn_states = rnn_states,
            init_sim_data = init_sim_data,
        )

    rollout_state = init_rollout_state()
    init_rnn_states = jnp.copy(rollout_state.rnn_states)
    
    def make_policy(policy_idx, init_rnd):
        variables = policy.init(
            init_rnd, random.PRNGKey(0), rollout_state.rnn_states,
            rollout_state.sim_data['obs'], 
            sample_actions = False,
            method='rollout')

        params = variables['params']
        params['backbone']['encoder']['net']['bias'] = jnp.array(
            policy_idx, dtype=jnp.int32)

        obs_preprocess = ObsPreprocessNoop().init(
            rollout_state.sim_data['obs'])
        obs_preprocess_state = obs_preprocess.init_state(
            rollout_state.sim_data['obs'])

        return PolicyState(
            apply_fn = policy.apply,
            rnn_reset_fn = policy.clear_recurrent_state,
            params = params,
            batch_stats = {},
            obs_preprocess = obs_preprocess,
            obs_preprocess_state = obs_preprocess_state,
        )

    rnd, rnd_init = random.split(rnd)

    make_policies = jax.jit(jax.vmap(make_policy))

    policy_states = make_policies(
        jnp.arange(rollout_cfg.total_num_policies),
        random.split(rnd_init, rollout_cfg.total_num_policies))

    return rnd, policy_states, rollout_state, rollout_cfg, init_obs, init_rnn_states

def verify_rollout_data(rollout_state, rollout_store, policy_states, init_obs,
                        init_rnn_states, num_steps, episode_len, batch_size):
    checkify.check(jnp.all(rollout_store['obs'][0] == init_obs),
                   "Init observation mismatch")

    actions_out = rollout_store['actions'][..., 0]
    assignments_out = rollout_store['actions'][..., 1]
    values_out = rollout_store['values'][..., 0]

    def gt_iter(global_step_idx, gt_state):
        cur_assignment = assignments_out[global_step_idx]
        is_done = jnp.logical_and(global_step_idx != 0,
                                  global_step_idx % episode_len == 0)

        episode_start_idx = (global_step_idx // episode_len) * episode_len

        checkify.check(
            jnp.all(assignments_out[episode_start_idx] == cur_assignment),
            "Policy assignment should not change during episode")

        policy_param = policy_states.params[
            'backbone']['encoder']['net']['bias'][cur_assignment][:, None]

        prev_actions = gt_state['actions'][global_step_idx]
        prev_values = gt_state['values'][global_step_idx]

        prev_values = jnp.where(is_done, jnp.zeros((), jnp.int32), prev_values)

        obs = prev_actions + 1
        step_actions = obs + policy_param

        step_values = prev_values + 2 * step_actions
        step_actions = step_actions + prev_values

        return gt_state.copy({
            'actions': gt_state['actions'].at[global_step_idx + 1].set(
                step_actions),
            'values': gt_state['values'].at[global_step_idx + 1].set(
                step_values),
        })

    def compute_gt_state(gt_state):
        return lax.fori_loop(0, num_steps, gt_iter, gt_state)

    gt_state = frozen_dict.freeze({
        'actions': jnp.zeros((num_steps + 1, batch_size, 1), dtype=jnp.int32),
        'values': jnp.zeros((num_steps + 1, batch_size, 1), dtype=jnp.int32),
    })

    gt_state = gt_state.copy({
        # Initial "actions" are just the initial obs - 1 to handle gt_iter
        # always adding one to compute the obs
        'actions': gt_state['actions'].at[0].set(init_obs - 1),
        'values': gt_state['values'].at[0].set(init_rnn_states),
    })

    gt_state = compute_gt_state(gt_state)

    gt_state = frozen_dict.freeze({
        'actions': gt_state['actions'][1:, ..., 0],
        # Critic adds 1
        'values': gt_state['values'][1:, ..., 0] + 1,
    })

    checkify.check(jnp.all(gt_state['values'] == values_out),
        "Value mismatch")
    checkify.check(jnp.all(gt_state['actions'] == actions_out),
        "Action mismatch")

    final_values = rollout_store['values'][-1, ...]
    final_rnn_states = rollout_state.rnn_states

    rnn_check = jnp.where(rollout_state.sim_data['dones'],
        jnp.zeros((), jnp.int32), final_values - 1)

    checkify.check(jnp.all(rnn_check == final_rnn_states), "RNN mismatch")

    checkify.check(jnp.all(rollout_store['actions'][..., 0:1] + 2 ==
        rollout_store['rewards']), "Reward mismatch")


def check_rollout_loop(
    rnd,
    num_steps,
    num_chunks, # Ignored
    episode_len,
    num_current_policies,
    num_past_policies,
    num_teams,
    team_size,
    batch_size,
    self_play,
    cross_play,
    past_play,
    policy_chunk_size_override = 0,
):
    rnd, policy_states, rollout_state, rollout_cfg, init_obs, init_rnn_states = fake_rollout_setup(
        rnd,
        num_steps,
        episode_len,
        num_current_policies,
        num_past_policies,
        num_teams,
        team_size,
        batch_size,
        self_play,
        cross_play,
        past_play,
        policy_chunk_size_override,
    )

    def check_assignments(assigns):
        assigns = assigns.reshape(-1, num_teams, team_size)

        checkify.check(
            jnp.all(assigns[:, :, 0:1] == assigns[:, :, 1:]),
            "All team members should be using the same policy")

        num_sp_matches = int(assigns.shape[0] * self_play)
        num_cp_matches = int(assigns.shape[0] * cross_play)
        num_pp_matches = int(assigns.shape[0] * past_play)

        sp_matches = assigns[:num_sp_matches]
        cp_matches = assigns[num_sp_matches:num_sp_matches + num_cp_matches]
        pp_matches = assigns[num_sp_matches + num_cp_matches:num_sp_matches + num_cp_matches + num_pp_matches]

        checkify.check(
            jnp.all(sp_matches[:, 0:1, :] == sp_matches[:, 1:, :]),
            "All teams in self play matches should have same policy")

        checkify.check(
            jnp.all(cp_matches[:, 0:1, :] != cp_matches[:, 1:, :]),
            "Cross play opponents should have different policies than team 0")

        checkify.check(
            jnp.all(pp_matches[:, 0:1, :] != pp_matches[:, 1:, :]),
            "Past play opponents should have different policies than team 0")

        checkify.check(jnp.all(jnp.logical_and(
                cp_matches[:, 1:, :] >= 0,
                cp_matches[:, 1:, :] < num_current_policies,
            )),
            "Invalid cross play policies")

        checkify.check(jnp.all(jnp.logical_and(
                pp_matches[:, 1:, :] >= num_current_policies,
                pp_matches[:, 1:, :] < num_current_policies + num_past_policies,
            )),
            "Invalid past play policies")

        sp_matches = sp_matches.reshape(
            num_current_policies, -1, *sp_matches.shape[1:])
        cp_matches = cp_matches.reshape(
            num_current_policies, -1, *cp_matches.shape[1:])
        pp_matches = pp_matches.reshape(
            num_current_policies, -1, *pp_matches.shape[1:])

        policy_indices = jnp.arange(num_current_policies).reshape(-1, 1, 1)

        checkify.check(
            jnp.all(sp_matches[:, :, 0, :] == policy_indices),
            "Incorrect self play train policies")

        checkify.check(
            jnp.all(cp_matches[:, :, 0, :] == policy_indices),
            "Incorrect cross play train policies")

        checkify.check(
            jnp.all(pp_matches[:, :, 0, :] == policy_indices),
            "Incorrect past play train policies")

    def post_inference_cb(step_idx, policy_obs, preprocessed_obs, policy_out,
                          reorder_state, rollout_store):
        obs, actions, values = reorder_state.to_sim(
            (preprocessed_obs, policy_out['actions'], policy_out['values']))
        
        return rollout_store.copy({
            'obs': rollout_store['obs'].at[step_idx].set(obs),
            'actions': rollout_store['actions'].at[step_idx].set(actions),
            'values': rollout_store['values'].at[step_idx].set(values),
        })

    def post_step_cb(step_idx, dones, rewards, reorder_state, rollout_store):
        return rollout_store.copy({
            'rewards': rollout_store['rewards'].at[step_idx].set(rewards),
        })

    def loop_wrapper(rollout_state, policy_states, init_obs, init_rnn_states):
        rollout_store = frozen_dict.freeze({
            'obs': jnp.zeros((num_steps, batch_size, 1), dtype=jnp.int32),
            'values': jnp.zeros((num_steps, batch_size, 1), dtype=jnp.int32),
            'actions': jnp.zeros((num_steps, batch_size, 2), dtype=jnp.int32),
            'rewards': jnp.zeros((num_steps, batch_size, 1), dtype=jnp.int32),
        })

        rollout_state, rollout_store = rollout_loop(
            rollout_state = rollout_state,
            policy_states = policy_states,
            rollout_cfg = rollout_cfg,
            num_steps = num_steps,
            post_inference_cb = post_inference_cb,
            post_step_cb = post_step_cb,
            cb_state = rollout_store,
            sample_actions = False,
            return_debug = False,
        )

        verify_rollout_data(rollout_state, rollout_store, policy_states,
            init_obs, init_rnn_states, num_steps, episode_len, batch_size)

        all_assignments = rollout_store['actions'][..., 1]
        jax.vmap(check_assignments)(all_assignments)

        return rollout_state, rollout_store

    loop_wrapper = jax.jit(checkify.checkify(loop_wrapper), donate_argnums=0)

    err, (rollout_state, rollout_store) = loop_wrapper(
        rollout_state, policy_states, init_obs, init_rnn_states)
    err.throw()

    return rollout_state, rollout_store
    

def check_rollout_mgr(
    rnd,
    num_steps,
    num_chunks,
    episode_len,
    num_current_policies,
    num_past_policies,
    num_teams,
    team_size,
    batch_size,
    self_play,
    cross_play,
    past_play,
    policy_chunk_size_override = 0,
):
    train_cfg = TrainConfig(
        num_worlds = batch_size // (num_teams * team_size),
        num_agents_per_world = num_teams * team_size,
        num_updates = 0,
        steps_per_update = num_steps,
        lr = 0,
        algo = None,
        num_bptt_chunks = num_chunks,
        gamma = 1,
        seed = rnd,
        gae_lambda = 1,
        pbt = PBTConfig(
            num_teams = num_teams,
            team_size = team_size,
            num_train_policies = num_current_policies,
            num_past_policies = num_past_policies,
            past_policy_update_interval = 0,
            self_play_portion = self_play,
            cross_play_portion = cross_play,
            past_play_portion = past_play,
            rollout_policy_chunk_size_override = policy_chunk_size_override,
        ),
    )

    (rnd, policy_states, rollout_state,
     rollout_cfg, init_obs, init_rnn_states) = fake_rollout_setup(
        rnd,
        num_steps,
        episode_len,
        num_current_policies,
        num_past_policies,
        num_teams,
        team_size,
        batch_size,
        self_play,
        cross_play,
        past_play,
        policy_chunk_size_override,
    )

    rollout_mgr = RolloutManager(
        train_cfg = train_cfg,
        rollout_cfg = rollout_cfg,
        init_rollout_state = rollout_state,
    )

    rnd, pbt_rnd = random.split(rnd, 2)

    value_normalizer = EMANormalizer(1, jnp.int32, disable=True)

    train_states = PolicyTrainState(
        value_normalizer = value_normalizer,
        tx = None,
        value_normalizer_state = value_normalizer.init_estimates(init_obs),
        hyper_params = None,
        opt_state = None,
        scheduler = None,
        scaler = None,
        update_prng_key = None,
    )

    train_state_mgr = TrainStateManager(
        policy_states = policy_states,
        train_states = train_states,
        pbt_rng = pbt_rnd,
    )

    def collect_wrapper(rollout_state, train_state_mgr,
                        init_obs, init_rnn_states):
        metrics = rollout_mgr.add_metrics(train_cfg, FrozenDict())
        metrics = TrainingMetrics.create(train_cfg, metrics)

        train_state_mgr, rollout_state, rollout_data, metrics = rollout_mgr.collect(
            train_state_mgr, rollout_state, metrics)

        train_slice = lambda x: x[rollout_mgr._sim_to_train_idxs]

        sliced_rollout_state = jax.tree_map(train_slice, rollout_state)
        sliced_init_obs = jax.tree_map(train_slice, init_obs)
        sliced_init_rnns = jax.tree_map(train_slice, init_rnn_states)

        def verify_wrapper(policy_idx, rollout_policy_data,
                sliced_rollout_state, sliced_init_obs, sliced_init_rnns):
            per_policy_batch_size = rollout_mgr._num_train_agents_per_policy

            rollout_policy_data, rnn_start_states = rollout_policy_data.pop(
                'rnn_start_states')

            rnn_start_states = rnn_start_states.reshape(
                num_chunks, per_policy_batch_size, *rnn_start_states.shape[1:])

            # Need to invert the transformation applied by
            # RolloutManager._finalize_rollouts
            def txfm(x):
                pre_permute = x.reshape(num_chunks, per_policy_batch_size,
                    num_steps // num_chunks, *x.shape[2:])

                # [C, T/C, B, ...]
                orig = pre_permute.transpose(
                    0, 2, 1, *range(3, len(pre_permute.shape)))

                return orig.reshape(num_steps, per_policy_batch_size, *orig.shape[3:])

            rollout_policy_data = jax.tree_map(txfm, rollout_policy_data)

            verify_rollout_data(sliced_rollout_state, rollout_policy_data,
                policy_states, sliced_init_obs, sliced_init_rnns, num_steps,
                episode_len, per_policy_batch_size)

            all_assignments = rollout_policy_data['actions'][..., 1]
            checkify.check(jnp.all(all_assignments == policy_idx),
                "Mismatched policy index for train data")

            ref_rnn_states = jnp.concatenate([
                    sliced_init_rnns[None, ...],
                    rollout_policy_data['values'] - 1, # Critic adds one
                ], axis=0)

            ref_rnn_states = ref_rnn_states.at[1:].set(
                jnp.where(rollout_policy_data['dones'],
                          jnp.zeros_like(ref_rnn_states[1:]),
                          ref_rnn_states[1:]))

            ref_rnn_states = ref_rnn_states[
                jnp.arange(0, num_steps, num_steps // num_chunks)]

            checkify.check(jnp.all(rnn_start_states == ref_rnn_states),
                "Invalid RNN start states")

        verify_wrapper = jax.vmap(verify_wrapper)
        verify_wrapper(jnp.arange(num_current_policies),
            rollout_data.data, sliced_rollout_state,
            sliced_init_obs, sliced_init_rnns)

        return rollout_state, train_state_mgr, rollout_data

    collect_wrapper = jax.jit(
        checkify.checkify(collect_wrapper), donate_argnums=0)
    err, (rollout_state, train_state_mgr, rollout_data) = collect_wrapper(
        rollout_state, train_state_mgr, init_obs, init_rnn_states)
    err.throw()


def test_rollouts():
    keys = ['rnd', 'num_steps', 'num_chunks', 'episode_len',
            'num_current_policies', 'num_past_policies',
            'num_teams', 'team_size', 'batch_size',
            'self_play', 'cross_play', 'past_play']

    configs = [
        [5,   3,  1, 11,  1, 0, 2, 1,     4, 1.0,  0.0, 0.0],
        [5,   4,  2, 11,  1, 0, 2, 1,     4, 1.0,  0.0, 0.0],
        [5,  10,  2, 11,  4, 0, 2, 2,    32, 0.0,  1.0, 0.0],
        [5, 200,  1, 10, 16, 7, 2, 2, 16384, 0.0,  1.0, 0.0],
        [5, 200,  1, 15, 16, 7, 4, 2, 16384, 0.0,  1.0, 0.0],
        [7, 200,  1, 15, 16, 0, 4, 2,   128, 1.0,  0.0, 0.0],
        [7, 200,  1, 15, 16, 7, 4, 2,  1024, 0.0,  0.5, 0.5],
        [7, 200,  1, 15, 16, 7, 4, 2,  1024, 0.5, 0.25, 0.25],
        [7, 200,  1, 15, 16, 7, 4, 4,  1024, 0.5, 0.25, 0.25],
        [7, 200,  4, 15, 16, 7, 4, 4,  1024, 0.0,  0.0, 1.0],
        [7, 1000, 1, 15, 16, 7, 4, 4,  1024, 0.0,  0.0, 1.0],
        [7, 1000, 4, 15, 16, 7, 4, 4,  4096, 0.0,  1.0, 0.0],
    ]

    for args in configs:
        kwargs = {k: v for k, v in zip(keys, args)}
        print(kwargs)
        check_rollout_loop(**kwargs)
        check_rollout_mgr(**kwargs)


test_reorder_chunks1()
test_reorder_chunks2()
test_reorder_chunks3()
test_reorder_chunks4()

#test_init_matchmake1()
#test_init_matchmake2()

test_rollouts()
