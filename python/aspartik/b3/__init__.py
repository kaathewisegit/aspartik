from typing import Protocol, runtime_checkable

from .._aspartik_rust_impl._b3_rust_impl import (
    MCMC as MCMC,
    Clock as Clock,
    Proposal as Proposal,
)


@runtime_checkable
class Stateful(Protocol):
    """
    Epoch-versioned objects for use with MCMC

    All stateful objects handled by `MCMC` must conform to this protocol.
    During each step operators can edit objects using whatever APIs provided.
    Then, at the end of the step, `MCMC` calls `accept` if the move has been
    accepted, or `reject` otherwise.
    """

    def accept(self) -> None:
        """Accept changes made during the current step"""

    def reject(self) -> None:
        """Reject changes made during the current step

        This method must roll the state of the object back to how it was at the
        beginning of the MCMC step.
        """


@runtime_checkable
class Prior(Protocol):
    """
    Interface which describes all prior distributions

    A `Prior` object will be queried by `MCMC` on each step after the operator
    edits the state to get the prior probability of the new state.  It's the
    responsibility of the prior to track the stateful objects it's interested
    in, so they will typically be passed in the constructor.

    Variable priors, coalescents, birth-death models, and all other
    non-likelihood models implement this protocol.
    """

    def probability(self) -> float:
        """Calculates the log prior probability of the model state

        The return value must be a **natural logarithm** of the probability.

        `MCMC` will short-circuit and abort the move if `probability` returns a
        negative infinity.  This can be used to avoid expensive likelihood
        calculations for obviously invalid moves, like going out of variable
        bounds.
        """
        ...


class Operator(Protocol):
    """
    Objects which propose moves by editing state

    It is the responsibility of the objects to track the parts of the state it
    might want to edit.  Typically these objects will be passed in the
    constructor.
    """

    weight: float
    """Influences the probability of the operator being picked

    On each step `MCMC` picks a random operator from the list passed to it.
    It uses this value to weight them.  So, the larger it is, the more
    often the operator will be picked, and visa versa.  This value is read
    once on startup.  Therefore, if it's changed mid-execution the old
    cached value will still be used.
    """

    def propose(self) -> Proposal:
        """Proposes a new MCMC step

        It is presumed that the operator will store all the references to
        parameters and trees it wants to edit and will change them accordingly.
        If a move cannot be proposed for any reason `Proposal.Reject` should be
        returned.  `MCMC` will deal with rolling back the state.
        """
        ...


class Callback(Protocol):
    """
    Custom callbacks

    `b3` supports arbitrary logging and checks via this protocol.  The `call`
    function is passed a reference to the main `MCMC` object, so the callback
    can either take state variables in the constructor of fetch them via the
    `MCMC` attributes.

    The `call` function won't be called after each step for efficiency.  See
    [`every`](#Callback.every) for configuring how often the callback will be
    invoked.
    """

    every: int
    """How often this callback should be called

    The `MCMC` will call each callback object when `index % every` is 0.  This
    value is read once when MCMC is created, so if it's changed during
    execution, the old `every` value will continue to be used.
    """

    def call(self, mcmc: MCMC) -> None:
        """
        A custom operation

        Used by loggers and other periodic actions.
        """
        ...

    def finish(self, mcmc: MCMC) -> None:
        """
        A method which will be called after a run ends

        It's useful for finalizing logger actions, including flushing to files.
        This method has a default no-op implementation, so it's not necessary
        to implement it on a custom logger.
        """
        pass
