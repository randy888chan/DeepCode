import asyncio
from typing import List, Optional, Dict, Any
from mcp_agent.core.fastagent import FastAgent

class FastAgentNoDecorator:
    def __init__(self, name: str):
        self.agents = FastAgent(name=name)
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._chains: Dict[str, Dict[str, Any]] = {}
        self._parallels: Dict[str, Dict[str, Any]] = {}
        self._routers: Dict[str, Dict[str, Any]] = {}

    def add_agent(self, name: str, model: str, instruction: str, servers: List[str]) -> None:
        """Add an agent without using decorator."""
        self._agents[name] = {
            "name": name,
            "model": model,
            "instruction": instruction,
            "servers": servers
        }
        # Create the agent using the original FastAgent
        self.agents.agent(
            name=name,
            model=model,
            instruction=instruction,
            servers=servers
        )

    def add_chain(self, name: str, sequence: List[str], instruction: str, cumulative: bool = True) -> None:
        """Add a chain workflow without using decorator."""
        self._chains[name] = {
            "name": name,
            "sequence": sequence,
            "instruction": instruction,
            "cumulative": cumulative
        }
        # Create the chain using the original FastAgent
        self.agents.chain(
            name=name,
            sequence=sequence,
            instruction=instruction,
            cumulative=cumulative
        )

    def add_parallel(self, name: str, fan_out: List[str], fan_in: str, instruction: str, include_request: bool = True) -> None:
        """Add a parallel workflow without using decorator."""
        self._parallels[name] = {
            "name": name,
            "fan_out": fan_out,
            "fan_in": fan_in,
            "instruction": instruction,
            "include_request": include_request
        }
        # Create the parallel workflow using the original FastAgent
        self.agents.parallel(
            name=name,
            fan_out=fan_out,
            fan_in=fan_in,
            instruction=instruction,
            include_request=include_request
        )

    def add_router(self, name: str, agents: List[str], model: str, use_history: bool = False, human_input: bool = False) -> None:
        """Add a router without using decorator."""
        self._routers[name] = {
            "name": name,
            "agents": agents,
            "model": model,
            "use_history": use_history,
            "human_input": human_input
        }
        # Create the router using the original FastAgent
        self.agents.router(
            name=name,
            agents=agents,
            model=model,
            use_history=use_history,
            human_input=human_input
        )

    async def run(self):
        """Run the agent system."""
        return await self.agents.run()

    def get_agent(self, name: str) -> Any:
        """Get an agent by name."""
        return getattr(self.agents, name)

    def get_chain(self, name: str) -> Dict[str, Any]:
        """Get a chain workflow by name."""
        return self._chains.get(name)

    def get_parallel(self, name: str) -> Dict[str, Any]:
        """Get a parallel workflow by name."""
        return self._parallels.get(name)

    def get_router(self, name: str) -> Dict[str, Any]:
        """Get a router by name."""
        return self._routers.get(name) 