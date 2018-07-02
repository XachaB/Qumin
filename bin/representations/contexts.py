# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""author: Sacha Beniamine.

This module implements patterns' contexts, which are series of phonological restrictions."""

from itertools import zip_longest
from representations.segments import Segment, _CharClass, restore, restore_string
from representations.quantity import one, optional, some, kleenestar, Quantity, quantity_largest, quantity_sum
from representations.alignment import align_right, align_left, align_levenshtein_multi

def restore_segment_shortest(segment):
    """Restore segment to the shortest of either the original character or its feature list."""
    if segment:
        return min([restore(segment), Segment.get(segment).shorthand], key=len)
    else:
        return segment

def _align_edges(*args, debug=False, **kwargs):
    """Align at both edges.

    Aligns an equal number of characters left and right, and all the rest in the center.
    A tuple of ("",Quantity(min_center_len,max_center_len)) is added at the center
    to ensure the correctness of future merges.
    """
    lengths = [len(x) for x in args]
    minl = min(lengths)
    if all([l == lengths[0] for l in lengths]):
        return list(zip(*args))
    else:
        cut = minl // 2

    left = []
    right = []
    center = set()
    center_lens = []
    for i, element in enumerate(args):
        l = len(element)
        left.append(element[:cut])
        right.append(element[l-cut:])
        center.update(element[cut:l-cut])
        center_lens.append(l-(2*cut))

    if center:
        center.add(("", Quantity(min(center_lens), max(center_lens))))

    if debug:
        print("Aligned center:",list(zip(*left)),"\n\t",[tuple(center)],"\n\t", list(zip(*right)))
    return list(zip(*left)) + [tuple(center)] + list(zip(*right))

class _ContextMember(object):
    """One part of a Context, between the edges of the word and/or a blank."""

    def __init__(self,elements,opt=False,blank=False):
        self.blank = blank
        if elements ==[] or elements[-1] == "{}":
            self.blank = True
            elements = elements[:-1]
        self.restrictions = elements
        self.opt = False #opt
        self.empty = self.restrictions == []

    def __len__(self):
        return len(self.restrictions)

    def __getitem__(self,i):
        return self.restrictions[i]

    def __str__(self):
        return self.to_str(mode=2)

    def __eq__(self,other):
        return repr(self) == repr(other)

    def __repr__(self):
        return self.to_str(mode=0)

    def to_str(self,mode=2):
        ## Format modes:
        ## str (0):      ([Ei]...) - with parenthesis, regex
        ## restored (1):  [ei]...
        ## display (2): shortest(restored,[+syll, -open])
        ## features (3): [+syll, -open]
        def to_features(segment):
            if segment:
                return Segment.get(segment).shorthand
            return segment

        format_modes = [str,restore,restore_segment_shortest,to_features]
        format_blanks = ["{}","_","_","_"]
        format_segment = format_modes[mode]
        blankchar = format_blanks[mode]
        formatted = ""
        if mode == 0 or (self.opt and not self.empty):
            formatted += "("

        for segment, quantifier in self.restrictions:
            if segment != "":
                formatted += format_segment(segment)
            formatted += str(quantifier)

        if self.opt and not self.empty:
            formatted += ")"+str(optional)
        elif mode == 0:
            formatted += ")"
        if self.blank:
            formatted += blankchar
        return formatted


    def simplify(self):
        """ Reduce a context in place for indexes i-1 to i.
        Simplifies a context's regex.

        A sequence of two identical segments with quantifiers $Q_{1}$ and $Q_{2}$
        is simplified as follows : $aQ_{1}aQ_{2} = aQ_{3}$
        for example  [aei]?[aei]+ is simplifiedto [aei]+

        If the result is one segment long, then if the context member is optional,
        the quantifier is reduced too.
        for example ([aei]+)? is simplified to [aei]*

        ================= === === === ===
         Q_1 / Q_2             ?   +   *
        ================= === === === ===
           (one)           +   +   +   +
         ? (optional)      +   *   +   *
         + (some)          +   +   +   +
         * (kleenestar)    +   *   +   *
        ================= === === === ===
        """
        equal_reduction = {
                            (one, one): some,
                            (one, optional): some,
                            (one, some): some,
                            (one, kleenestar): some,
                            (optional, optional): kleenestar,
                            (optional, some): some,
                            (optional, kleenestar): kleenestar,
                            (some, some): some,
                            (some, kleenestar): some,
                            (optional, one): some,
                            (some, one): some,
                            (kleenestar, one): some,
                            (some, optional): some,
                            (kleenestar, optional): kleenestar,
                            (kleenestar, some): some,
                            (kleenestar, kleenestar): kleenestar,
                        }
        reduction = {(optional, some): some,
                     (optional, kleenestar): kleenestar,
                     (kleenestar, some): some,
                     (some,some): some,
                     (kleenestar, kleenestar): kleenestar}
        i = 1 # starts at 1, because at each step, we simplify from i-1 to i
        while i < len(self.restrictions):

            (s1,q1),(s2,q2) = self.restrictions[i-1:i+1]

            # print("Quantifiers [{}]~[{}]".format(repr(q1),repr(q2)),"in reduction ?",(q1, q2) in reduction)
            # print(s1,"<",s2,"?",Segment.get(s1) < Segment.get(s2))
            if s1 == s2 and (q1, q2) in equal_reduction:
                self.restrictions[i-1:i+1] = [(s2, equal_reduction[(q1,q2)])]
                if i > 1:
                    i -= 1
            elif (q1, q2) in reduction and Segment.get(s1) < Segment.get(s2):
                self.restrictions[i-1:i+1] = [(s2, reduction[(q1,q2)])]
                if i > 1:
                    i -= 1
            else:
                i += 1
        if len(self.restrictions) == 1 and self.opt:
            self.opt = False
            s,q = self.restrictions[0]
            self.restrictions[0] = (s, q & optional)

class Context(object):
    """Context for an alternation pattern"""

    def __init__(self,segments,ortho=False):
        if segments == []:
            self.elements = _ContextMember([],blank=True)
        elif type(segments[0]) is _ContextMember:
            self.elements = segments
        else:
            prev = 0
            self.elements = []
            for i in range(len(segments)):
                if segments[i] == "{}":
                    self.elements.append(_ContextMember(segments[prev:i+1]))
                    prev = i+1
            if prev < len(segments):
                self.elements.append(_ContextMember(segments[prev:]))

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self,i):
        return self.elements[i]

    def __repr__(self):
        return self.to_str(mode=0)

    def __str__(self):
        return self.to_str(mode=2)

    def to_str(self,mode=2):
        return "".join(member.to_str(mode=mode) for member in self.elements)

    @classmethod
    def _align(cls,contexts,debug=False):
        ''' Align contexts segment by segment in order to merge (generator).
        '''
        l = max(len(c) for c in contexts)

        leftblank = False
        align_funcs = {(False,True):align_right,
                       (True,False):align_left,
                        (False,False):align_levenshtein_multi, # This is an identity pattern, alignment is arbitrary.
                       (True,True):_align_edges}

        for i in range(l):
            restrictions = []
            opt = False
            rightblank = False
            for c in contexts:
                if i >= len(c) or c[i].empty:
                    opt = True
                    restrictions.append([("", optional)])
                else:
                    rightblank = rightblank or c[i].blank # shouldn't even need to accumulate it
                    opt = opt or c[i].opt
                    if c[i].restrictions not in restrictions: # don't put in several times the same context member !
                        restrictions.append(c[i].restrictions)
            align_func = align_funcs[(leftblank,rightblank)]

            yield (align_func(*restrictions,fillvalue=("", optional),debug=debug),opt,rightblank)
            leftblank = rightblank

    @classmethod
    def merge(cls,contexts,debug=False):
        """ Merge contexts to generalize them.

        Merge contexts and combine their restrictions into a new context.

        Arguments:
            contexts: iterable of Contexts.

        Returns:
            a merged context
        """
        new_context = []
        if debug:
            print("Alignigning",contexts)
            print(list(cls._align(contexts, debug=debug)))
        for group,opt,blank in cls._align(contexts, debug=debug):
            context_members = []

            buffer_sources = [] # for debug purposes
            buffer_segments = []
            buffer_quantities = []
            for aligned_segments in group:
                has_empty = ('', optional) in aligned_segments

                # Add to buffer if there is an empty segment
                if has_empty:
                    segments, quantities = zip(*aligned_segments)
                    quantity = quantity_largest(quantities)
                    buffer_segments.extend([s for s in segments if s])
                    buffer_quantities.append(quantity)
                    buffer_sources.append(aligned_segments)
                # Use buffer
                else:
                    if buffer_segments:
                        s1,q1 = Segment.intersect(*buffer_segments), quantity_sum(buffer_quantities)
                        if debug:
                            print(buffer_sources,"->",s1,q1)
                        context_members.append((s1,q1))
                        # re-init buffer
                        buffer_segments = []
                        buffer_quantities = []
                        buffer_sources = []
                    # Not buffer
                    segments, quantities = zip(*aligned_segments)
                    segment = Segment.intersect(*[s for s in segments if s])
                    quantity = quantity_largest(quantities)
                    if debug:
                        print(aligned_segments,"->",(segment,quantity))
                    context_members.append((segment, quantity))
            if buffer_segments:
                context_members.append((Segment.intersect(*buffer_segments),
                                        quantity_sum(buffer_quantities)))
            new_context_member = _ContextMember(context_members,blank=blank, opt=opt)
            new_context_member.simplify()
            new_context.append(new_context_member)

        return Context(new_context)
