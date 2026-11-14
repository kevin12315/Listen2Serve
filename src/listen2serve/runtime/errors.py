"""Runtime errors distinguished by the controller."""


class EndpointSessionClosed(RuntimeError):
    """An established realtime session was closed by the server.

    Connection-establishment errors remain ordinary RuntimeError instances;
    this exception identifies interrupted calls after a successful connection.
    """


class AgentResponseTimeout(RuntimeError):
    """The current response exceeded its existing wait limit; end the session."""
