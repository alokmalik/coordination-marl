import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, n_agents=3, use_dense_rewards=False, shuffle_landmarks=False, color_objects=True,
                   small_agents=False):
        world = World()
        world.use_dense_rewards = use_dense_rewards
        self.shuffle_landmarks = shuffle_landmarks
        self.color_objects = color_objects
        self.small_agents = small_agents

        colors=self.agent_colors(n_agents)

        # set any world properties first
        num_agents = n_agents
        num_landmarks = n_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            if i==0:
                agent.max_speed=.2
            else:
                agent.max_speed=1
            agent.color=colors[i]
            agent.clip_positions = np.array([[-world.scale, -world.scale], [world.scale, world.scale]])
            agent.is_colliding = {other_agent.name:False for other_agent in world.agents if agent is not other_agent}
            if not self.small_agents:
                agent.size *= 3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            if not self.small_agents:
                landmark.size *= 0.5
            else:
                landmark.size *= 2

        # make initial conditions
        self.reset_world(world)

        return world

    def agent_colors(self,n_agents):
        if n_agents<8:
            def DecimalToBinary(num,binary):
                if num >= 1:
                    DecimalToBinary(num // 2,binary)
                    binary.append(num%2)
                return binary
            color_matrix=[]
            for i in range(n_agents):
                binary=DecimalToBinary(i,[])
                while len(binary)<3:
                    binary.insert(0,0)
                color_matrix.append(binary)
        else:
            color_matrix=np.random.random(n_agents,3)
            color_matrix[:,0]=np.array([0,0,1]) 
        return color_matrix

    def reset_world(self, world):
        colors=self.agent_colors(len(world.agents))
        #colors = [np.array([0.8, 0., 0.]), np.array([0., 0.8, 0.]), np.array([0., 0., 0.8])]
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if not self.color_objects:
                agent.color = np.array([0.8, 0.5, 0.2])
                agent._color = agent.color
            else:
                agent.color = colors[i]
                agent._color = agent.color
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if not self.color_objects:
                landmark.color = np.array([0.75, 0.75, 0.75])
            else:
                landmark.color = colors[i]
        # set random initial states
        for agent in world.agents:
            pos=np.random.uniform(-world.scale, +world.scale, world.dim_p)
            pos[0]=.95
            agent.state.p_pos = pos
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if self.shuffle_landmarks:
                agent.point_of_vue = np.random.permutation(len(world.landmarks))

        for i, landmark in enumerate(world.landmarks):
            pos=np.random.uniform(-world.scale, +world.scale, world.dim_p)
            landmark.state.p_pos = pos
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        raise NotImplemented

    def is_collision(self, agent1, agent2):
        if agent1 is agent2 or not agent1.collide or not agent2.collide:
            return False

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size

        if dist < dist_min:
            agent1.is_colliding[agent2.name] = True
            agent2.is_colliding[agent1.name] = True
            return True

        else:
            agent1.is_colliding[agent2.name] = False
            agent2.is_colliding[agent1.name] = False
            return False

    def count_collisions(self, agent, world):
        n_collisions = 0
        for a_i in world.agents:
            for a_j in world.agents:
                if self.is_collision(a_i, a_j):
                    n_collisions += 0.5

        # Sets agent's color based on whether it is colliding or not
        if any(agent.is_colliding.values()):
            agent.color = np.array([0.2, 0.2, 0.2])
        else:
            agent.color = agent._color

        return n_collisions


    def sparse_reward(self, agent, world):
        rew = 0
        #if all agents are occupying a landmark then agent gets +100 reward 
        for l in world.landmarks:
            if not self.small_agents:
                agents_in = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) < a.size ** 2 for a in world.agents]
            else:
                agents_in = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) < l.size**2 for a in world.agents]
            rew += 100. if sum(agents_in)==len(world.agents) else 0.

        return rew

    def dense_reward(self, agent, world):
        #the agent gets reward proportional to the nearest
        rew=0
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        d=min(dists)
        
        if d<agent.size:
            rew = (3-min(dists))**2 * 0.1
        return rew
        
         

    def reward(self,agent,world):
        personal_rewards =[]

        for a in world.agents:
            personal_rewards.append(self.dense_reward(a,world))
        
        personal_rewards=np.array(personal_rewards)

        #authoritarian network
        n=len(world.agents)
        network=np.zeros((n,n))
        np.fill_diagonal(network,1)
        network[:,0]=1
        
        reward_type='multiplicative'
        agent_i=int(agent.name[-1])
        #multiplicative reward:
        if reward_type=='multiplicative':
            net=network[agent_i,:]
            rew=1
            for j,power in enumerate(net):
                rew*=personal_rewards[j]**power
        else:    
            #additive reward
            personal_rewards= np.matmul(personal_rewards,network)

            agent_i=int(agent.name[-1])

            rew=personal_rewards[agent_i]
        
        return rew



    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_col=[]
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_col.append(entity.color)
        # communication and position of all other agentsof all other agents
        comm = []
        other_pos = []
        
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if self.shuffle_landmarks:
            entity_pos = np.array(entity_pos)[agent.point_of_vue]
            entity_pos = list(entity_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
