.. ## Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======
Logging
=======

Logging is done through Axom's `SLIC <https://axom.readthedocs.io/en/develop/axom/slic/docs/sphinx/index.html>`_
component. SLIC provides a lot of configurable logging functionality which we have consolidated the header
``src/common/Logger.hpp`` and implemented in ``src/common/Logger.cpp``.

.. note::
  On parallel runs, messages can be out of order if flush is not called often enough.

Logging Streams
---------------

SLIC has a concept of logging streams.  Logging streams controls the following:

 * How each message is formatted. More info `here <https://axom.readthedocs.io/en/develop/axom/slic/docs/sphinx/sections/architecture.html#log-message-format>`_ .
 * Where each messages are output, such as ``std::cout``, ``std::cerr``, or to a file stream.
 * Logic for handling and filtering of messages, based on message level or content.

Serac creates the following logging streams under different conditions:


   * GenericOutputStream

     * Serial
     * Debug and info messages to ``std::cout``
     * Warning and error messages to ``std::cerr``
     * Logs all messages directly to given streams.

   * LumberjackStream

     * Parallel and SERAC_USE_LUMBERJACK is true
     * Debug and info messages to ``std::cout``
     * Warning and error messages to ``std::cerr``
     * Passes messages to rank 0 and filters duplicates before outputting.

   * SynchronizedStream

     * Parallel and SERAC_USE_LUMBERJACK is false
     * Debug and info messages to ``std::cout``
     * Warning and error messages to ``std::cerr``
     * Prints messages on one rank at a time each flush.

Message Levels
--------------

SLIC has 4 message levels to help indicate the important of messages. Descriptions are as follows:

 * Debug - messages that help debugging runs, only on when ``SERAC_DEBUG`` is defined
 * Info - basic informational messages
 * Warning - message indicating that something has gone wrong but not enough to end the simulation
 * Error - message indicating a non-recoverable error has occurred

Before every warning and error, ``serac::logger::Flush()`` should be called to clarify what happened
leading up to the message.  After an error has occured ``serac::ExitGracefully(bool error=false)`` should
be called.

Logging Messages
----------------

SLIC provides many helper macros that assist in logging messages. Here is a list of them but more information
can be found `here <https://axom.readthedocs.io/en/develop/axom/slic/docs/sphinx/sections/appendix.html#slic-macros-used-in-axom>`_ :

 * ``SLIC_DEBUG(msg)``
 * ``SLIC_DEBUG_IF(expression, msg)``
 * ``SLIC_DEBUG_RANK0(rank, msg)``
 * ``SLIC_INFO(msg)``
 * ``SLIC_INFO_IF(expression, msg)``
 * ``SLIC_INFO_RANK0(rank, msg)``
 * ``SLIC_WARNING(msg)``
 * ``SLIC_WARNING_IF(expression, msg)``
 * ``SLIC_WARNING_RANK0(rank, msg)``
 * ``SLIC_ERROR(msg)``
 * ``SLIC_ERROR_IF(expression, msg)``
 * ``SLIC_ERROR_RANK0(rank, msg)``


.. note::
  Macros with RANK0 in the name are not true SLIC macros but are defined by Serac.

