
"""Mock tools for Meeting Notes Agent. Supports Arena fault injection."""

import asyncio
import time
import logging
from typing import Any

from app.models import Fault

log = logging.getLogger('tools')

# Registry of allowed tools
TOOL_REGISTRY = {
    'extract_action_items',
    'create_calendar_event',
    'request_clarification',
}


async def execute_tool(
    tool_name: str,
    arguments: dict,
    fault: Fault,
    attempt: int = 1,
) -> dict[str, Any]:
    """Execute a tool with fault injection support for Arena testing."""

    start = time.perf_counter()

    # --- Fault injection layer (Arena controls this) ---
    if fault.type == 'tool_timeout' and attempt == 1:
        log.warning('FAULT INJECTED: tool_timeout on %s', tool_name)
        await asyncio.sleep(45)  # exceeds run_timeout_seconds
        return {'status': 'timeout', 'error': 'Tool timed out'}

    if fault.type == 'malformed_tool_output' and attempt == 1:
        log.warning('FAULT INJECTED: malformed_tool_output on %s', tool_name)
        # Return something that breaks expected structure
        return {'status': 'success', 'data': None, 'unexpected_field': [1, 2, 3]}

    # --- Normal tool execution ---
    try:
        if tool_name == 'extract_action_items':
            result = _extract_action_items(arguments)
        elif tool_name == 'create_calendar_event':
            result = _create_calendar_event(arguments)
        elif tool_name == 'request_clarification':
            result = _request_clarification(arguments)
        else:
            result = {'status': 'rejected', 'error': f'Unknown tool: {tool_name}'}
    except Exception as e:
        log.error('Tool %s raised: %s', tool_name, e)
        result = {'status': 'exception', 'error': str(e)}

    elapsed_ms = (time.perf_counter() - start) * 1000
    result['_latency_ms'] = elapsed_ms
    return result


def _extract_action_items(arguments: dict) -> dict:
    """Mock extraction — in production this would be an LLM sub-call."""
    notes = arguments.get('notes_text', '')
    if not notes:
        return {'status': 'error', 'error': 'No notes_text provided'}
    return {
        'status': 'success',
        'data': {'parsed_length': len(notes), 'note': 'Extraction handled by LLM decision layer'},
    }


def _create_calendar_event(arguments: dict) -> dict:
    """Mock sandbox calendar creation. No real API call."""
    title = arguments.get('title', '')
    date = arguments.get('date', '')
    assignee = arguments.get('assignee', '')

    if not title:
        return {'status': 'error', 'error': 'Missing title'}
    if not date:
        return {'status': 'error', 'error': 'Missing date'}

    return {
        'status': 'success',
        'data': {
            'event_id': f'mock-{hash(title + date) % 10000:04d}',
            'title': title,
            'date': date,
            'assignee': assignee or 'unassigned',
            'sandbox': True,
        },
    }


def _request_clarification(arguments: dict) -> dict:
    """Return clarification question."""
    question = arguments.get('question', 'Could you provide more details?')
    return {'status': 'success', 'data': {'question': question}}
