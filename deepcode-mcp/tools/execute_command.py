import asyncio
import sys
from mcp.server import InitializationOptions
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

async def serve():
    server = Server("exec-server")

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "exec_command":
            cmd = arguments.get("command")
            if not cmd:
                return [{"type": "text", "text": "No command provided"}]
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await proc.communicate()
            return [
                {"type": "text", "text": out.decode() + err.decode()}
            ]
        else:
            return [{"type": "text", "text": "Unknown tool"}]

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)

if __name__ == "__main__":
    asyncio.run(serve())