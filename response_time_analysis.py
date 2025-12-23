#!/usr/bin/env python3
"""
Script to analyze and visualize response time differences between user turns and bot turns
across multiple conversation.json files.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import argparse


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO format timestamp string to datetime object."""
    return datetime.fromisoformat(timestamp_str)


def calculate_response_times(conversation_data: dict) -> List[Tuple[int, float]]:
    """
    Calculate time differences in milliseconds between user_turn and the following bot_turn.
    
    Returns a list of tuples: (turn_number, time_difference_ms)
    """
    messages = conversation_data.get("messages", [])
    response_times = []
    
    for i, msg in enumerate(messages):
        # Look for user turns
        if msg.get("role") == "user":
            user_timestamp = parse_timestamp(msg["timestamp"])
            user_turn = msg.get("turn", "")
            
            # Extract turn number from "user_turn_X"
            try:
                turn_num = int(user_turn.split("_")[-1])
            except (ValueError, IndexError):
                continue
            
            # Find the next bot turn
            for j in range(i + 1, len(messages)):
                next_msg = messages[j]
                if next_msg.get("role") == "assistant":
                    bot_timestamp = parse_timestamp(next_msg["timestamp"])
                    
                    # Calculate time difference in milliseconds
                    time_diff = (bot_timestamp - user_timestamp).total_seconds() * 1000
                    response_times.append((turn_num, time_diff))
                    break
    
    return response_times


def load_all_conversations(recordings_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load all conversation.json files from the recordings directory.
    
    Returns a dictionary: {session_id: [(turn_num, time_diff_ms), ...]}
    """
    conversations = {}
    recordings_path = Path(recordings_dir)
    
    if not recordings_path.exists():
        print(f"Error: Directory {recordings_dir} does not exist")
        return conversations
    
    for session_dir in sorted(recordings_path.iterdir()):
        if session_dir.is_dir():
            conversation_file = session_dir / "conversation.json"
            if conversation_file.exists():
                try:
                    with open(conversation_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    session_id = data.get("session_id", session_dir.name)
                    response_times = calculate_response_times(data)
                    
                    if response_times:  # Only add if we have valid data
                        conversations[session_id] = response_times
                        print(f"Loaded {len(response_times)} turn pairs from {session_id}")
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error loading {conversation_file}: {e}")
    
    return conversations


def plot_response_times(conversations: Dict[str, List[Tuple[int, float]]], 
                        output_file: str = None,
                        show_plot: bool = True):
    """
    Plot response times for all conversations with different colors.
    
    X-axis: Turn number
    Y-axis: Response time in milliseconds
    """
    if not conversations:
        print("No conversations to plot")
        return
    
    # Set up the plot with a dark theme for better aesthetics
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Generate distinct colors using a colormap
    colors = plt.cm.Set2(np.linspace(0, 1, len(conversations)))
    
    # Custom color palette for better distinction
    custom_colors = [
        '#FF6B6B',  # Coral Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Sky Blue
        '#96CEB4',  # Sage Green
        '#FFEAA7',  # Pale Yellow
        '#DDA0DD',  # Plum
        '#98D8C8',  # Mint
        '#F7DC6F',  # Soft Yellow
        '#BB8FCE',  # Lavender
        '#85C1E9',  # Light Blue
        '#F8B500',  # Golden
        '#00CED1',  # Dark Turquoise
    ]
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']
    
    legend_handles = []
    
    for idx, (session_id, response_times) in enumerate(conversations.items()):
        color = custom_colors[idx % len(custom_colors)]
        marker = markers[idx % len(markers)]
        
        turns = [rt[0] for rt in response_times]
        times = [rt[1] for rt in response_times]
        
        # Plot line with markers
        line, = ax.plot(turns, times, 
                       color=color, 
                       marker=marker,
                       markersize=8,
                       linewidth=2,
                       alpha=0.8,
                       label=session_id)
        legend_handles.append(line)
        
        # Add scatter points for emphasis
        ax.scatter(turns, times, color=color, s=60, alpha=0.9, zorder=5, edgecolors='white', linewidths=0.5)
    
    # Styling
    ax.set_xlabel('Turn Number', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Response Time (milliseconds)', fontsize=14, fontweight='bold', color='white')
    ax.set_title('User Turn to Bot Turn Response Time Analysis', fontsize=18, fontweight='bold', color='white', pad=20)
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(handles=legend_handles, 
              loc='upper right', 
              fontsize=10,
              framealpha=0.8,
              facecolor='#2d2d2d',
              edgecolor='white',
              title='Session ID',
              title_fontsize=12)
    
    # Set axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Add background color gradient effect
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#16213e')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Plot saved to {output_file}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def print_statistics(conversations: Dict[str, List[Tuple[int, float]]]):
    """Print summary statistics for each conversation."""
    print("\n" + "="*70)
    print("RESPONSE TIME STATISTICS (milliseconds)")
    print("="*70)
    
    for session_id, response_times in conversations.items():
        times = [rt[1] for rt in response_times]
        if times:
            print(f"\n📊 Session: {session_id}")
            print(f"   Total turns analyzed: {len(times)}")
            print(f"   Min response time:    {min(times):.2f} ms")
            print(f"   Max response time:    {max(times):.2f} ms")
            print(f"   Average response time: {np.mean(times):.2f} ms")
            print(f"   Median response time:  {np.median(times):.2f} ms")
            print(f"   Std deviation:        {np.std(times):.2f} ms")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Analyze response times between user and bot turns')
    parser.add_argument('--recordings-dir', '-r', 
                        default='recordings',
                        help='Path to the recordings directory (default: recordings)')
    parser.add_argument('--output', '-o',
                        default='response_time_graph.png',
                        help='Output file for the graph (default: response_time_graph.png)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (only save to file)')
    parser.add_argument('--sessions', '-s', nargs='*',
                        help='Specific session IDs to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Load conversations
    print(f"Loading conversations from: {args.recordings_dir}")
    conversations = load_all_conversations(args.recordings_dir)
    
    # Filter sessions if specified
    if args.sessions:
        conversations = {k: v for k, v in conversations.items() if k in args.sessions}
    
    if not conversations:
        print("No valid conversations found!")
        return
    
    # Print statistics
    print_statistics(conversations)
    
    # Plot
    plot_response_times(conversations, 
                        output_file=args.output,
                        show_plot=not args.no_show)


if __name__ == "__main__":
    main()

