import time
import asyncio
import numpy as np

'''
4. Coherence
    4.(1) first token:   self.queue.arr[idx] += tokens
    4.(2) other token:   self.queue.shift(n)
    4.(3) max_tokens in range would be update upon new req arrivals
'''

class FixedCircularQueue:
    # revealing arr, head to public
    def __init__(self, length):
        self.length = length
        self.arr = np.zeros(length, dtype=int)
        
        self.head = 0 

    # Fixed length, new elements set to 0
    def shift(self, n):
        new_head = (self.head + n) % self.length

        # set [old_head, new_head-1]* to zeros
        if new_head > self.head:
            self.arr[self.head:new_head] = 0
        else:
            self.arr[self.head:] = 0
            self.arr[:new_head] = 0
        
        self.head = new_head



# the effective lookahead distance for the forecast info (max_tokens)
LOOKAHEAD_RANGE = 200

class Lookahead:
    def __init__(self, instance_id, max_context_length=4096):
        self.queue = FixedCircularQueue(max_context_length * 2)
        self.lock = asyncio.Lock()

        self.instance_id = instance_id
        
        self.lookahead_requests = {}
        self.global_current_timestep = 0
        self.lookahead_max_tokens = 0

    
    # generated_tokens == 1
    def request_first_token(self, request_id, prompt_len, predicted_output_len):
        # concurrent safe
        current_timestep = self.global_current_timestep
        head = self.queue.head  

        self.lookahead_requests[request_id] = {
            "prompt_len": prompt_len,
            # "predicted_output_len": predicted_output_len,
            "generated_tokens": 1,
            "start_timestep": current_timestep,
            "end_timestep": current_timestep + predicted_output_len - 1
        }

        # Update queue (Might need to lock)
        tokens = prompt_len + 1
        for i in range(predicted_output_len):
            idx = (head + i) % self.queue.length
            self.queue.arr[idx] += tokens
            tokens += 1

        # Update lookahead_max_tokens (Might need to lock)
        if head + LOOKAHEAD_RANGE <= self.queue.length:
            self.lookahead_max_tokens = max(self.queue.arr[head : head+LOOKAHEAD_RANGE])
        else:
            max_end = max(self.queue.arr[head : self.queue.length])
            max_start = max(self.queue.arr[: (head+LOOKAHEAD_RANGE) % self.queue.length])
            self.lookahead_max_tokens = max(max_end, max_start)
        

    # generated_tokens >= 2
    def request_other_token(self, request_id, generated_tokens):
        request = self.lookahead_requests[request_id]  # TODO: Check if KeyError

        req_current_timestep = request["start_timestep"] + generated_tokens - 1
        global_current_timestep = self.global_current_timestep  # concurrent safe

        # Update global_current_timestep and queue (Would need to lock to maintain the head ~ global_current_timestep)
        if req_current_timestep > global_current_timestep:
            forward_steps = req_current_timestep - global_current_timestep
            self.global_current_timestep = req_current_timestep
            self.queue.shift(forward_steps)
        
        # Req: late stop
        EXTEND_BASE = 10
        EXTEND_TOKENS = 30
        if req_current_timestep > request["end_timestep"] + EXTEND_BASE:
            # TODO: Can calibrate the req_current_timestep here (preempted request re-enter) 

            # Extend even if req_current_timestep << global_current_timestep 
            req_end_timestep = req_current_timestep + EXTEND_TOKENS
            request["end_timestep"] = req_end_timestep  # concurrent safe

            # Update queue [global_current_timestep, new_end_timestep]
            global_current_timestep = self.global_current_timestep
            if req_end_timestep > global_current_timestep: # No need for base comparison, extend if possible
                head = self.queue.head
                tokens = generated_tokens + request["prompt_len"] + global_current_timestep - req_current_timestep
                for i in range(req_end_timestep - global_current_timestep):
                    idx = (head + i) % self.queue.length
                    self.queue.arr[idx] += tokens
                    tokens += 1

                # Rely on high qps, NO need to update lookahead_max_tokens
            
        request["generated_tokens"] = generated_tokens


    def request_end(self, request_id):
        ROLLBACK_BASE = 10

        if request_id in self.lookahead_requests:
            # Req: early stop
            current_timestep = self.global_current_timestep  # concurrent safe
            head = self.queue.head
            request = self.lookahead_requests[request_id]
            if request["end_timestep"] - current_timestep > ROLLBACK_BASE:
                # update queue [current_timestep, end_timestep] 
                tokens = current_timestep - request["start_timestep"] + 1 + request["prompt_len"]
                for i in range(request["end_timestep"] - current_timestep + 1):
                    idx = (head + i) % self.queue.length
                    self.queue.arr[idx] -= tokens
                    tokens += 1

                # print(f"Warning: request_id {request_id} ends before predicted_output_len and need to rollaback {rollback_steps} steps")

            del self.lookahead_requests[request_id]
    
    def get_lookahead_max_tokens(self):
        return self.lookahead_max_tokens

    def get_lookahead_200(self):
        # return self.queue.arr[(self.queue.head + 200) % self.queue.length]
        return self.queue.arr[(self.queue.head + 400) % self.queue.length]


# # Test
# if __name__ == "__main__":
#     lookahead = Lookahead(4096)
