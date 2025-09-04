"""
Global timing tracker for LangChain retrieval components.

Tracks execution sequence and timing for better visibility into retrieval pipeline performance.
"""

import time
import logging
from typing import Dict, List, Optional
from contextlib import contextmanager
from threading import Lock

logger = logging.getLogger(__name__)

class TimingTracker:
    """Global tracker for retrieval component timing."""
    
    def __init__(self):
        self.timings: List[Dict] = []
        self.current_query: Optional[str] = None
        self.lock = Lock()
    
    def start_query(self, query: str):
        """Start tracking a new query."""
        with self.lock:
            self.current_query = query
            self.timings = []
            self.timings.append({
                "event": "query_start",
                "query": query,
                "timestamp": time.time(),
                "type": "marker"
            })
    
    def record_timing(self, component: str, operation: str, start_time: float, end_time: float, 
                     metadata: Optional[Dict] = None):
        """Record timing for a component operation."""
        with self.lock:
            duration = end_time - start_time
            timing_entry = {
                "component": component,
                "operation": operation,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "query": self.current_query,
                "type": "timing",
                "metadata": metadata or {}
            }
            self.timings.append(timing_entry)
    
    def get_timings_summary(self) -> str:
        """Get a formatted summary of all timings."""
        if not self.timings:
            return "No timings recorded"
        
        # Sort by start time
        sorted_timings = sorted([t for t in self.timings if t["type"] == "timing"], 
                              key=lambda x: x["start_time"])
        
        if not sorted_timings:
            return "No component timings recorded"
        
        total_duration = sorted_timings[-1]["end_time"] - sorted_timings[0]["start_time"]
        
        summary = f"Retrieval Pipeline Timing (Total: {total_duration:.3f}s)\n"
        summary += "=" * 50 + "\n"
        
        for timing in sorted_timings:
            duration = timing["duration"]
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            metadata_str = ""
            if timing["metadata"]:
                meta_parts = []
                for k, v in timing["metadata"].items():
                    if isinstance(v, (int, float)):
                        meta_parts.append(f"{k}: {v}")
                    elif isinstance(v, str):
                        meta_parts.append(f"{k}: {v}")
                if meta_parts:
                    metadata_str = f" ({', '.join(meta_parts)})"
            
            summary += f"{timing['component']} - {timing['operation']}: {duration:.3f}s ({percentage:.1f}%){metadata_str}\n"
        
        return summary
    
    def log_summary(self):
        """Log the timing summary."""
        if self.timings:
            logger.info("\n" + self.get_timings_summary())

# Global instance
timing_tracker = TimingTracker()

@contextmanager
def track_component_timing(component: str, operation: str, metadata: Optional[Dict] = None):
    """Context manager for tracking component timing."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        timing_tracker.record_timing(component, operation, start_time, end_time, metadata)

def start_query_tracking(query: str):
    """Start tracking a new query."""
    timing_tracker.start_query(query)

def get_timing_summary() -> str:
    """Get current timing summary."""
    return timing_tracker.get_timings_summary()

def log_timing_summary():
    """Log current timing summary."""
    timing_tracker.log_summary()
