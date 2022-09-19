from gym.envs.registration import register

register(
    id='Follow-The-Leader-v0', 
    entry_point='follow_the_leader_gym.envs:FollowTheLeaderEnv',
    )
