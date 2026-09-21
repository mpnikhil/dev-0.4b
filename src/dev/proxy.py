"""A stdio MCP proxy. Only tools explicitly wired through this process are affected."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import jsonschema
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from .inference import Predictor
from .shim import RawStore, compress, shortlist


def make_server(args):
    predictor = Predictor(args.checkpoint, args.device)
    store = RawStore(args.store)
    sessions = {}

    @asynccontextmanager
    async def lifespan(server):
        params = StdioServerParameters(command=args.upstream[0], args=args.upstream[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                sessions["upstream"] = session
                sessions["lock"] = asyncio.Lock()
                yield {}

    server = FastMCP("dev", lifespan=lifespan, instructions=(
        "Use tools_discover to obtain complete tool schemas, then tools_invoke with the chosen name and arguments. "
        "Compressed output is an excerpt, not a complete result. Use outputs_expand with raw_ref to recover exact text. "
        "This proxy is experimental. It does not grant permission to perform an action."
    ))

    async def catalog():
        found, cursor = [], None
        while True:
            page = await sessions["upstream"].list_tools(cursor=cursor)
            found.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                return found

    @server.tool(annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    async def tools_discover(query: str, top_k: int = 3, all_tools: bool = False) -> dict:
        """Find tools for a request; returns complete schemas. all_tools bypasses filtering."""
        async with sessions["lock"]:
            tools = [tool.model_dump(mode="json", exclude_none=True) for tool in await catalog()]
            selected = shortlist(tools, query, predictor, top_k,
                                 args.experimental and not all_tools)
        return {"tools": selected, "catalog_size": len(tools), "returned": len(selected),
                "experimental": args.experimental}

    # Generic dispatch cannot faithfully project each upstream tool's permission category.
    # Conservatively mark it as potentially destructive; do not recommend blanket approval.
    @server.tool(annotations=types.ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True))
    async def tools_invoke(name: str, arguments: dict, objective: str) -> types.CallToolResult:
        """Invoke a discovered tool by exact name and validated arguments. May have side effects."""
        async with sessions["lock"]:
            catalog_by_name = {tool.name: tool for tool in await catalog()}
            if name not in catalog_by_name:
                raise ValueError("Unknown upstream tool")
            jsonschema.validate(arguments, catalog_by_name[name].inputSchema)
            result = await sessions["upstream"].call_tool(name, arguments)
            # Structured/multimodal/error results pass through unchanged, including isError.
            if result.isError or result.structuredContent is not None or any(block.type != "text" for block in result.content):
                return result
            blocks = []
            for block in result.content:
                packed = compress(block.text, objective, predictor, store, args.max_chars, args.experimental)
                blocks.append(block.model_copy(update={"text": packed["text"]}))
            return result.model_copy(update={"content": blocks})

    @server.tool(annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def outputs_expand(raw_ref: str, start: int = 0, end: int | None = None) -> str:
        """Read exact cached original text, optionally by zero-based character range [start,end)."""
        return store.get(raw_ref, start, end)

    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/bootstrap")
    p.add_argument("--store", default=".cache/raw-results")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--experimental", action="store_true", help="Enable unvalidated filtering; default passes through")
    p.add_argument("--upstream", nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    if not args.upstream:
        p.error("--upstream requires an executable and optional arguments")
    make_server(args).run(transport="stdio")


if __name__ == "__main__":
    main()
