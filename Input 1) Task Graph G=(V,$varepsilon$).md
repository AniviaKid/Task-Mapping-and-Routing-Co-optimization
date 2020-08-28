**Input**: 1) Task Graph *G*=(v,$\varepsilon$) ; 2) NoC: *P*=n×n ; 3) Mapping : *M* (v $\rightarrow$ *P*) 

**Output**: Routing set *R*

1:  initialize an empty set *R* , sort out-links in $\varepsilon$ by priority

3:  **for** $e \in \varepsilon$ **do** 

4: 	 add $e$ into *R* 

5: 	 **while NOT** the route of $e$ is calculated: 

6: 	 	send current state to **Actor-Critic** (state is current position of data to be transmitted in the mesh)

7: 	 	get how to transmit (along X or along Y) at this step

8: 	 	fill the remaining route of $e$ by XY routing

9: 	 	send *R* to **onlineCompute** to get reward (**ONLY** based on the route of **Actor-Critic** output)

10:  	  move to the next state

11:    **end while**

12: **end for**

