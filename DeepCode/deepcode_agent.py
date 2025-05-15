import asyncio
from mcp_agent.core.fastagent import FastAgent
from Deepcode.agents.code_agents import agents
from Deepcode.ui.logo import print_logo

async def main() -> None:
    print_logo()
    async with agents.run() as agent:
        await agent.prompt("Deepcodeworkflow")

if __name__ == "__main__":
    asyncio.run(main())  # type: ignore
